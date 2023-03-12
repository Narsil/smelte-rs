#![feature(test)]

extern crate test;
use smelt::ops::normalize;
use smelt::tensor::OwnedTensor;
use smelt::tensor::TensorMut;
use test::{black_box, Bencher};

#[bench]
fn bench_normalize(b: &mut Bencher) {
    let mut tensor = OwnedTensor::zeros(vec![110, 768]);
    let mut max = vec![0.0; 768];
    let mut var = vec![0.0; 768];
    b.iter(|| {
        let tensor = black_box(&mut tensor);
        normalize(tensor, &mut max, &mut var, 1e-5).unwrap();
    });
}
