use crate::Metric;
use std::marker::PhantomData;

#[derive(Default, Debug, Clone)]
pub struct EuclideanMetric<T> {
    _phantom: PhantomData<T>,
}

impl<T> Metric for EuclideanMetric<T>
where
    for<'a> &'a T: IntoIterator<Item = &'a f64>,
    T: 'static,
{
    type PointType = T;

    fn distance(&self, p1: &Self::PointType, p2: &Self::PointType) -> f64 {
        p1.into_iter()
            .zip(p2.into_iter())
            .fold(0.0, |acc, (l, r)| acc + (l - r).powf(2.0))
            .sqrt()
    }
}

#[derive(Debug, Clone)]
pub struct WeightedEuclideanMetric<T> {
    weights: T,
}

impl<T> WeightedEuclideanMetric<T> {
    pub fn new(weights: T) -> Self {
        Self { weights }
    }
}

impl<T> Metric for WeightedEuclideanMetric<T>
where
    for<'a> &'a T: IntoIterator<Item = &'a f64>,
    T: 'static,
{
    type PointType = T;

    fn distance(&self, p1: &Self::PointType, p2: &Self::PointType) -> f64 {
        p1.into_iter()
            .zip(p2.into_iter())
            .zip((&self.weights).into_iter())
            .fold(0.0, |acc, ((l, r), w)| acc + w * (l - r).powf(2.0))
            .sqrt()
    }
}
