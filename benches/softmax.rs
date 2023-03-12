#![feature(test)]

extern crate test;
use smelt::ops::softmax;
use smelt::tensor::OwnedTensor;
use smelt::tensor::TensorMut;
use test::{black_box, Bencher};

#[bench]
fn bench_softmax(b: &mut Bencher) {
    let mut tensor = OwnedTensor::zeros(vec![12, 110, 110]);
    let mut max = vec![0.0; 12 * 110];
    b.iter(|| {
        let tensor = black_box(&mut tensor);
        softmax(tensor, &mut max).unwrap();
    });
}
