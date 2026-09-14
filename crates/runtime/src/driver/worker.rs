use super::{DriverConfig, DriverError, Interest, Shared, ShutdownReply, Submission};
use crate::{Engine, Event, RequestId};
use flume::{Receiver, Sender};
use std::sync::Arc;
use std::sync::atomic::Ordering;

struct Route {
    request: RequestId,
    interest: Arc<Interest>,
    events: Sender<Event>,
}

struct Exit<'a> {
    shared: Arc<Shared>,
    done: Sender<()>,
    result: Result<(), DriverError>,
    submissions: &'a Receiver<Submission>,
    shutdown: &'a Receiver<ShutdownReply>,
}

impl Drop for Exit<'_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.result = Err(DriverError::WorkerPanicked);
            *self
                .shared
                .failure
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner) =
                Some(DriverError::WorkerPanicked);
        }
        self.shared
            .stop(self.result.clone().err().unwrap_or(DriverError::Closed));
        *self
            .shared
            .exit
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(self.result.clone());
        // Setting stopped/exit first excludes queue insertion under the matching
        // locks. Flume retains queued values while senders live even after the
        // last receiver is dropped; explicitly drop every pending acknowledgement.
        while self.submissions.try_recv().is_ok() {}
        while self.shutdown.try_recv().is_ok() {}
        let _ = self.done.try_send(());
    }
}

pub(super) fn run(
    engine: Engine,
    config: DriverConfig,
    shared: Arc<Shared>,
    submissions: &Receiver<Submission>,
    shutdown: &Receiver<ShutdownReply>,
    wake: &Receiver<()>,
    done: Sender<()>,
) {
    // Exit is dropped after the inner engine, including during unwind. A panic
    // disconnects output; clients treat an unexplained disconnect as owner panic.
    let mut exit = Exit {
        shared,
        done,
        result: Err(DriverError::WorkerPanicked),
        submissions,
        shutdown,
    };
    let result = {
        let mut engine = engine;
        drive(
            &mut engine,
            config,
            &exit.shared,
            submissions,
            shutdown,
            wake,
        )
    };
    exit.result = result;
}

fn drive(
    engine: &mut Engine,
    config: DriverConfig,
    shared: &Shared,
    submissions: &Receiver<Submission>,
    shutdown: &Receiver<ShutdownReply>,
    wake: &Receiver<()>,
) -> Result<(), DriverError> {
    let mut routes = Vec::with_capacity(config.max_requests);
    loop {
        // Never drain after checking work and before sleeping: a racing notifier
        // would then have its only persistent signal erased.
        let _ = wake.try_recv(); // capacity one; do not chase racing producers
        if shared.final_stop.load(Ordering::SeqCst) {
            abandon(engine, &mut routes, submissions);
            return engine.shutdown().map_err(DriverError::Owner);
        }
        if let Ok(reply) = shutdown.try_recv() {
            abandon(engine, &mut routes, submissions);
            let result = engine.shutdown().map_err(DriverError::Owner);
            let succeeded = result.is_ok();
            let _ = reply.try_send(result);
            if succeeded {
                return Ok(());
            }
            // Closed to new work, but the same thread/engine retains retry ownership.
            continue;
        }
        if shared.check_open().is_err() {
            abandon(engine, &mut routes, submissions);
            let _ = wake.recv();
            continue;
        }
        let mut progress = accept(engine, &mut routes, submissions, config.max_requests);
        progress |= deliver(engine, &mut routes);
        if shared.check_open().is_err() {
            continue;
        }
        let step = match engine.step() {
            Ok(step) => step,
            Err(error) => {
                shared.stop(DriverError::Owner(error));
                abandon(engine, &mut routes, submissions);
                continue;
            }
        };
        progress |= deliver(engine, &mut routes);
        if progress || step.submitted || step.completed {
            continue;
        }
        let status = engine.status();
        let needs_poll = status.in_flight
            || (status.waiting > 0
                && status.active_sequences < engine.config().max_active_requests);
        #[cfg(test)]
        if let Some((entered, resume)) = shared.before_wait.lock().unwrap().take() {
            entered.send(()).unwrap();
            resume.recv().unwrap();
        }
        if needs_poll {
            let _ = wake.recv_timeout(config.poll_interval);
        } else {
            let _ = wake.recv();
        }
    }
}

fn accept(
    engine: &mut Engine,
    routes: &mut Vec<Route>,
    submissions: &Receiver<Submission>,
    limit: usize,
) -> bool {
    let mut progress = false;
    for _ in 0..limit {
        let Ok(submission) = submissions.try_recv() else {
            break;
        };
        progress = true;
        if submission.interest.discard.load(Ordering::SeqCst) {
            continue;
        }
        match engine.enqueue_retained(submission.input, submission.interest.clone()) {
            Ok(request) => {
                if submission.accepted.try_send(Ok(request)).is_err() {
                    engine.discard(request);
                } else {
                    routes.push(Route {
                        request,
                        interest: submission.interest,
                        events: submission.events,
                    });
                }
            }
            Err(error) => {
                let _ = submission
                    .accepted
                    .try_send(Err(DriverError::Enqueue(error)));
            }
        }
    }
    progress
}

fn deliver(engine: &mut Engine, routes: &mut Vec<Route>) -> bool {
    let mut progress = false;
    let mut index = 0;
    while index < routes.len() {
        let route = &routes[index];
        if route.interest.discard.load(Ordering::SeqCst) || route.events.is_disconnected() {
            engine.discard(route.request);
            routes.swap_remove(index);
            progress = true;
            continue;
        }
        if route.interest.cancel.swap(false, Ordering::SeqCst) {
            // A terminal mailbox may outlive its execution identity. Unknown is
            // the only cancellation error and means the terminal already settled.
            let _ = engine.cancel(route.request);
            progress = true;
        }
        let mut finished = false;
        while !route.events.is_full() {
            let Some(event) = engine.pop_event_for(route.request) else {
                break;
            };
            progress = true;
            finished = matches!(event, Event::Finished { .. });
            // Only this worker sends. A concurrent receive can only add space;
            // therefore a failed send is disconnection, not capacity overflow.
            if route.events.try_send(event).is_err() {
                engine.discard(route.request);
                finished = true;
            }
            if finished {
                break;
            }
        }
        if finished {
            routes.swap_remove(index);
        } else {
            index += 1;
        }
    }
    progress
}

fn abandon(engine: &mut Engine, routes: &mut Vec<Route>, submissions: &Receiver<Submission>) {
    for route in routes.drain(..) {
        engine.discard(route.request);
    }
    // Dropping the ack sender wakes pending submissions with the shared owner
    // error; dropping each envelope returns its permit when its future also drops.
    while submissions.try_recv().is_ok() {}
}
