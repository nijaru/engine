//! Bounded request mailboxes. Execution uses stable internal slots; only the
//! public request-specific drain hashes a `RequestId`. A mailbox outlives its
//! execution slot until terminal delivery. The intrusive ready list never
//! accumulates stale entries and does not scan other clients' events.
use crate::{Event, RequestId};
use std::collections::{HashMap, VecDeque};

/// Internal lease. Never exposed to clients or retained after terminal delivery.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OutputId(usize);

struct Mailbox {
    request: RequestId,
    events: VecDeque<Event>,
    reserved: usize,
    terminal: bool,
    discarded: bool,
    linked: bool,
    prev: Option<usize>,
    next: Option<usize>,
}

pub(crate) struct Output {
    mailboxes: Vec<Option<Mailbox>>,
    free: Vec<usize>,
    requests: HashMap<RequestId, usize>,
    head: Option<usize>,
    tail: Option<usize>,
    buffered: usize,
    reserved: usize,
    limit: usize,
    per_request: usize,
}

impl Output {
    pub(crate) fn new(limit: usize, per_request: usize, requests: usize) -> Self {
        Self {
            mailboxes: Vec::with_capacity(requests),
            free: Vec::with_capacity(requests),
            requests: HashMap::with_capacity(requests),
            head: None,
            tail: None,
            buffered: 0,
            reserved: 0,
            limit,
            per_request,
        }
    }

    pub(crate) fn register(&mut self, request: RequestId) -> OutputId {
        let index = self.free.pop().unwrap_or_else(|| {
            self.mailboxes.push(None);
            self.mailboxes.len() - 1
        });
        self.mailboxes[index] = Some(Mailbox {
            request,
            events: VecDeque::with_capacity(self.per_request.min(4)),
            reserved: 0,
            terminal: false,
            discarded: false,
            linked: false,
            prev: None,
            next: None,
        });
        assert!(self.requests.insert(request, index).is_none());
        OutputId(index)
    }

    pub(crate) fn len(&self) -> usize {
        self.buffered
    }

    pub(crate) fn credits(&self, id: OutputId) -> usize {
        let mailbox = self.mailboxes[id.0].as_ref().expect("live mailbox");
        if mailbox.terminal || mailbox.discarded {
            return 0;
        }
        (self.limit - self.buffered - self.reserved)
            .min(self.per_request - mailbox.events.len() - mailbox.reserved)
    }

    pub(crate) fn reserve(&mut self, id: OutputId, count: usize) {
        assert!(
            count <= self.credits(id),
            "output credits were not available"
        );
        self.mailboxes[id.0]
            .as_mut()
            .expect("live mailbox")
            .reserved += count;
        self.reserved += count;
    }

    pub(crate) fn unreserve(&mut self, id: OutputId) {
        let mailbox = self.mailboxes[id.0].as_mut().expect("live mailbox");
        self.reserved -= mailbox.reserved;
        mailbox.reserved = 0;
    }

    pub(crate) fn can_finish(&self, id: OutputId) -> bool {
        let mailbox = self.mailboxes[id.0].as_ref().expect("live mailbox");
        mailbox.discarded || self.credits(id) > 0
    }

    pub(crate) fn push_to(&mut self, id: OutputId, event: Event) {
        let mailbox = self.mailboxes[id.0].as_ref().expect("live mailbox");
        debug_assert_eq!(mailbox.request, event.request());
        if mailbox.discarded {
            // Discard also cancels execution. Only its settled terminal may arrive;
            // it retires the mailbox without consuming delivery capacity.
            assert!(matches!(event, Event::Finished { .. }));
            self.reclaim(id.0);
            return;
        }
        assert!(self.credits(id) > 0, "output exceeded its reservation");
        let mailbox = self.mailboxes[id.0].as_mut().expect("live mailbox");
        mailbox.terminal = matches!(event, Event::Finished { .. });
        mailbox.events.push_back(event);
        self.buffered += 1;
        self.link(id.0);
    }

    /// Round-robin between ready requests; order within each request is exact.
    pub(crate) fn pop(&mut self) -> Option<Event> {
        let index = self.head?;
        let event = self.pop_index(index)?;
        if self.mailboxes[index]
            .as_ref()
            .is_some_and(|mailbox| mailbox.linked)
        {
            self.unlink(index);
            self.link(index);
        }
        Some(event)
    }

    pub(crate) fn pop_for(&mut self, id: RequestId) -> Option<Event> {
        let index = *self.requests.get(&id)?;
        self.pop_index(index)
    }

    /// Suppress present and future delivery without relinquishing reservations
    /// belonging to an in-flight submission. Terminal publication retires the box.
    pub(crate) fn discard(&mut self, request: RequestId) {
        let Some(&index) = self.requests.get(&request) else {
            return;
        };
        self.unlink(index);
        let mailbox = self.mailboxes[index].as_mut().expect("live mailbox");
        self.buffered -= mailbox.events.len();
        mailbox.events.clear();
        mailbox.discarded = true;
        if mailbox.terminal {
            self.reclaim(index);
        }
    }

    fn reclaim(&mut self, index: usize) {
        let mailbox = self.mailboxes[index].take().expect("terminal mailbox");
        assert!(mailbox.events.is_empty() && mailbox.reserved == 0 && !mailbox.linked);
        self.requests.remove(&mailbox.request);
        self.free.push(index);
    }

    fn pop_index(&mut self, index: usize) -> Option<Event> {
        let mailbox = self.mailboxes[index].as_mut().expect("live mailbox");
        let event = mailbox.events.pop_front()?;
        self.buffered -= 1;
        if mailbox.events.is_empty() {
            self.unlink(index);
        }
        if matches!(event, Event::Finished { .. }) {
            self.reclaim(index);
        }
        Some(event)
    }

    fn link(&mut self, index: usize) {
        let mailbox = self.mailboxes[index].as_mut().expect("live mailbox");
        if mailbox.linked {
            return;
        }
        mailbox.linked = true;
        mailbox.prev = self.tail;
        mailbox.next = None;
        if let Some(tail) = self.tail {
            self.mailboxes[tail].as_mut().expect("ready tail").next = Some(index);
        } else {
            self.head = Some(index);
        }
        self.tail = Some(index);
    }

    fn unlink(&mut self, index: usize) {
        let mailbox = self.mailboxes[index].as_mut().expect("live mailbox");
        if !mailbox.linked {
            return;
        }
        mailbox.linked = false;
        let prev = mailbox.prev.take();
        let next = mailbox.next.take();
        if let Some(prev) = prev {
            self.mailboxes[prev]
                .as_mut()
                .expect("ready predecessor")
                .next = next;
        } else {
            self.head = next;
        }
        if let Some(next) = next {
            self.mailboxes[next].as_mut().expect("ready successor").prev = prev;
        } else {
            self.tail = prev;
        }
    }

    #[cfg(test)]
    fn push(&mut self, event: Event) {
        self.push_to(OutputId(self.requests[&event.request()]), event);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FinishReason, Usage};

    fn token(id: RequestId, token: u32) -> Event {
        Event::Token { request: id, token }
    }
    fn finish(id: RequestId) -> Event {
        Event::Finished {
            request: id,
            reason: FinishReason::Length,
            usage: Usage::default(),
        }
    }

    #[test]
    fn discard_retains_in_flight_credits_until_settlement() {
        let mut output = Output::new(4, 4, 2);
        let a = RequestId(1);
        let b = RequestId(2);
        let a_box = output.register(a);
        let b_box = output.register(b);
        output.reserve(a_box, 3);
        output.discard(a);
        assert_eq!(output.credits(a_box), 0);
        assert!(output.can_finish(a_box));
        assert_eq!(output.credits(b_box), 1);
        assert_eq!(output.reserved, 3);
        output.unreserve(a_box);
        output.push_to(a_box, finish(a));
        assert_eq!(output.credits(b_box), 4);
        assert!(!output.requests.contains_key(&a));
        assert_eq!(output.len(), 0);
    }

    #[test]
    fn discarded_terminal_does_not_need_capacity_held_by_a_peer() {
        let mut output = Output::new(2, 2, 2);
        let abandoned = RequestId(1);
        let peer = RequestId(2);
        let abandoned_box = output.register(abandoned);
        output.register(peer);
        output.push(token(peer, 1));
        output.push(token(peer, 2));
        output.discard(abandoned);
        output.push_to(abandoned_box, finish(abandoned));
        assert_eq!(output.len(), 2);
        assert!(!output.requests.contains_key(&abandoned));
        assert_eq!(output.pop(), Some(token(peer, 1)));
        assert_eq!(output.pop(), Some(token(peer, 2)));
    }

    #[test]
    fn discard_unlinks_ready_mailboxes_without_changing_peer_order() {
        let mut output = Output::new(8, 4, 3);
        let ids = [RequestId(1), RequestId(2), RequestId(3)];
        for id in ids {
            output.register(id);
            output.push(token(id, 7));
            output.push(finish(id));
        }
        output.discard(ids[1]);
        output.discard(ids[0]);
        assert_eq!(output.pop(), Some(token(ids[2], 7)));
        assert_eq!(output.pop(), Some(finish(ids[2])));
        assert!(output.requests.is_empty());
        assert!(output.head.is_none() && output.tail.is_none());
        assert_eq!(output.len(), 0);
    }

    #[test]
    fn reservations_and_drain_are_isolated_by_request() {
        let a = RequestId(1);
        let b = RequestId(2);
        let mut output = Output::new(6, 3, 2);
        let a_box = output.register(a);
        let b_box = output.register(b);
        output.reserve(a_box, 3);
        output.reserve(b_box, 2);
        assert_eq!(output.credits(a_box), 0);
        assert_eq!(output.credits(b_box), 1);
        output.unreserve(a_box);
        output.push(token(a, 1));
        output.push(token(a, 2));
        output.push(finish(a));
        output.unreserve(b_box);
        output.push(token(b, 3));
        output.push(finish(b));
        assert_eq!(output.pop_for(b), Some(token(b, 3)));
        assert_eq!(output.pop_for(b), Some(finish(b)));
        assert_eq!(output.len(), 3);
        assert_eq!(output.pop(), Some(token(a, 1)));
        assert_eq!(output.pop(), Some(token(a, 2)));
        assert_eq!(output.pop(), Some(finish(a)));
        assert!(output.requests.is_empty());
        assert!(output.head.is_none() && output.tail.is_none());
    }

    #[test]
    fn mixed_draining_keeps_the_ready_list_bounded() {
        let a = RequestId(1);
        let b = RequestId(2);
        let c = RequestId(3);
        let mut output = Output::new(8, 4, 3);
        for id in [a, b, c] {
            output.register(id);
        }
        for _ in 0..1000 {
            for id in [a, b, c] {
                output.push(token(id, 1));
            }
            assert_eq!(output.pop_for(b), Some(token(b, 1)));
            assert_eq!(output.pop(), Some(token(a, 1)));
            assert_eq!(output.pop_for(c), Some(token(c, 1)));
            assert!(output.pop().is_none());
            assert!(output.head.is_none() && output.tail.is_none());
        }
        for id in [a, b, c] {
            output.push(finish(id));
        }
        for id in [b, c, a] {
            assert_eq!(output.pop_for(id), Some(finish(id)));
        }
        assert!(output.requests.is_empty());
    }

    #[test]
    fn aggregate_pop_is_fair_and_preserves_each_request_order() {
        let a = RequestId(1);
        let b = RequestId(2);
        let mut output = Output::new(8, 4, 2);
        for id in [a, b] {
            output.register(id);
            output.push(token(id, 1));
            output.push(token(id, 2));
        }
        assert_eq!(output.pop(), Some(token(a, 1)));
        assert_eq!(output.pop(), Some(token(b, 1)));
        assert_eq!(output.pop(), Some(token(a, 2)));
        assert_eq!(output.pop(), Some(token(b, 2)));
    }
}
