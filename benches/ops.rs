#![feature(test)]

extern crate test;
use smelt::ops::{apply, gelu, normalize, softmax};
use smelt::tensor::OwnedTensor;
use smelt::tensor::TensorMut;
use test::{black_box, Bencher};

#[bench]
fn bench_gelu(b: &mut Bencher) {
    let mut tensor = OwnedTensor::zeros(vec![110, 768]);
    b.iter(|| {
        let tensor = black_box(&mut tensor);
        apply(tensor, gelu)
    });
}

#[bench]
fn bench_normalize(b: &mut Bencher) {
    let mut tensor = OwnedTensor::zeros(vec![110, 768]);
    b.iter(|| {
        let tensor = black_box(&mut tensor);
        normalize(tensor, 1e-5).unwrap();
    });
}

#[bench]
fn bench_softmax(b: &mut Bencher) {
    let mut tensor = OwnedTensor::zeros(vec![12, 110, 110]);
    b.iter(|| {
        let tensor = black_box(&mut tensor);
        softmax(tensor).unwrap();
    });
}
