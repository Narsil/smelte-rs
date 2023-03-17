#![feature(test)]

extern crate test;
use smelt::cpu::f32::{apply, gelu, normalize, softmax, Tensor};
use test::{black_box, Bencher};

#[bench]
fn bench_gelu(b: &mut Bencher) {
    let mut tensor = Tensor::zeros(vec![110, 768]);
    b.iter(|| {
        let tensor = black_box(&mut tensor);
        apply(tensor, gelu)
    });
}

#[bench]
fn bench_normalize(b: &mut Bencher) {
    let mut tensor = Tensor::zeros(vec![110, 768]);
    b.iter(|| {
        let tensor = black_box(&mut tensor);
        normalize(tensor, 1e-5).unwrap();
    });
}

#[bench]
fn bench_softmax(b: &mut Bencher) {
    let mut tensor = Tensor::zeros(vec![12, 110, 110]);
    b.iter(|| {
        let tensor = black_box(&mut tensor);
        softmax(tensor).unwrap();
    });
}
