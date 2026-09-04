from pathlib import Path

p = Path("crates/nvidia/examples/qwen_serving_bench.rs")
s = p.read_text()
old = '''    println!(
        "  aggregate throughput: {:.2} tok/s",
        generated_tokens as f64 / elapsed.as_secs_f64()
    );
'''
new = '''    let generated_tokens_f64 = f64::from(
        u32::try_from(generated_tokens)
            .map_err(|_| "generated-token count exceeds benchmark reporting range".to_owned())?,
    );
    println!(
        "  aggregate throughput: {:.2} tok/s",
        generated_tokens_f64 / elapsed.as_secs_f64()
    );
'''
if old not in s:
    raise SystemExit("throughput conversion target not found")
s = s.replace(old, new, 1)
old = '''    let total = values.iter().map(Duration::as_secs_f64).sum::<f64>();
    let mean = Duration::from_secs_f64(total / values.len() as f64);
'''
new = '''    let total = values.iter().map(Duration::as_secs_f64).sum::<f64>();
    let count = u32::try_from(values.len()).expect("benchmark sample count fits u32");
    let mean = Duration::from_secs_f64(total / f64::from(count));
'''
if old not in s:
    raise SystemExit("duration count conversion target not found")
s = s.replace(old, new, 1)
p.write_text(s)
