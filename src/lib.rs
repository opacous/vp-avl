#![feature(test)]
#![feature(let_chains)]

mod iter;
mod metric;

use iter::*;

use metric::*;

extern crate test;

use replace_with::replace_with_or_abort;
use std::{collections::BinaryHeap, marker::PhantomData};

pub trait Metric {
    type PointType;
    fn distance(&self, p1: &Self::PointType, p2: &Self::PointType) -> f64;
}

pub trait VpTreeObject: Sized {
    type PointType: PartialEq;
    fn location(&self) -> &Self::PointType;
    // used to roughly estimate size of n-d balls
    // essentially an optimization, will still work if wrong
    fn dimension(&self) -> usize {
        10
    }

    fn approx_halving_radius(&self) -> f64 {
        // using stirling approximation of gamma fn
        let dim = self.dimension() as f64;

        // volume of unit sphere
        let unit_volume = 1.0 / (dim * std::f64::consts::PI).sqrt()
            * (2.0 * std::f64::consts::PI * std::f64::consts::E / dim).powf(dim / 2.0);

        // radius of ball of volume half the unit sphere in dim-dimensional euclidean space
        let half_radius = (std::f64::consts::PI * dim).powf(1.0 / (2.0 * dim))
            * (dim / (2.0 * std::f64::consts::PI * std::f64::consts::E)).sqrt()
            * (0.5_f64 * unit_volume).powf(1.0 / dim);

        half_radius
    }
}

impl VpTreeObject for Vec<f64> {
    type PointType = Self;
    fn location(&self) -> &Self {
        self
    }
}

pub trait Storage: FromIterator<Self::DType> + IntoIterator<Item = Self::DType> {
    type DType;
    fn read(&self, index: usize) -> &Self::DType;
    fn write(&mut self, index: usize, value: Self::DType);
    fn replace(&mut self, index: usize, value: Self::DType) -> Self::DType;
    fn map_i<F: FnOnce(Self::DType) -> Self::DType>(&mut self, index: usize, op: F);
    fn size(&self) -> usize;
    fn push(&mut self, value: Self::DType);
    fn pop(&mut self) -> Option<Self::DType>;
    fn iter(&self) -> impl Iterator<Item = &Self::DType>;
}

impl<T> Storage for Vec<T> {
    type DType = T;

    fn read(&self, index: usize) -> &Self::DType {
        &self[index]
    }

    fn write(&mut self, index: usize, value: Self::DType) {
        self[index] = value;
    }

    fn replace(&mut self, index: usize, mut value: Self::DType) -> Self::DType {
        std::mem::swap(&mut value, &mut self[index]);
        value
    }

    fn map_i<F: FnOnce(Self::DType) -> Self::DType>(&mut self, index: usize, op: F) {
        replace_with_or_abort(&mut self[index], op)
    }

    fn size(&self) -> usize {
        self.len()
    }

    fn push(&mut self, value: Self::DType) {
        self.push(value)
    }

    fn pop(&mut self) -> Option<Self::DType> {
        self.pop()
    }

    fn iter(&self) -> impl Iterator<Item = &Self::DType> {
        self.into_iter()
    }
}

pub trait VpAvl: Sized {
    type Point: VpTreeObject;
    type PointMetric: Metric<PointType = <Self::Point as VpTreeObject>::PointType>;
    type NodeStorage: Storage<DType = Node>;
    type DataStorage: Storage<DType = Self::Point>;

    fn nodes(&self) -> &Self::NodeStorage;
    fn nodes_mut(&mut self) -> &mut Self::NodeStorage;
    fn data(&self) -> &Self::DataStorage;
    fn data_mut(&mut self) -> &mut Self::DataStorage;
    fn metric(&self) -> &Self::PointMetric;
    fn root(&self) -> usize;
    fn set_root(&mut self, new: usize);

    fn node_index_data(&self, node_index: usize) -> &Self::Point {
        &self.data().read(self.nodes().read(node_index).center)
    }

    fn bulk_build_indices(&mut self, root: usize, mut indices: Vec<usize>) {
        // #[cfg(test)]
        // let mut prior_indices = indices.clone();

        if indices.len() < 2 {
            // simpler case
            match indices.len() {
                0 => {
                    // leaf node
                    self.nodes_mut().map_i(root, |mut node| {
                        node.height = 0;
                        node.interior = None;
                        node.exterior = None;
                        node
                    });
                }
                1 => {
                    // still has one child

                    let exterior = indices.pop().unwrap();

                    let radius = self.metric().distance(
                        self.node_index_data(root).location(),
                        self.node_index_data(exterior).location(),
                    );
                    self.nodes_mut().map_i(root, |mut node| {
                        node.exterior = Some(exterior);
                        node.radius = radius;
                        node.height = 1;
                        node.interior = None;
                        node.parent = Some(root);
                        node
                    });

                    self.bulk_build_indices(self.nodes().read(root).exterior.unwrap(), indices)
                }
                _ => unreachable!(),
            }
            return;
        }

        let mut distances = Vec::with_capacity(indices.len());
        for index in indices.iter() {
            distances.push((
                *index,
                self.metric().distance(
                    self.node_index_data(root).location(),
                    self.node_index_data(*index).location(),
                ),
            ));
        }
        // sort indices by distance from root
        distances.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let partitions: Vec<&[(usize, f64)]> = distances
            .chunks(distances.len() / 2 + distances.len() % 2)
            .collect();
        let mut interior_indices: Vec<usize> = partitions[0].iter().map(|x| x.0).collect();
        let mut exterior_indices: Vec<usize> = partitions[1].iter().map(|x| x.0).collect();

        let min_exterior_distance = partitions[1].first().unwrap().1;

        let interior_center = interior_indices.pop();
        let exterior_center = exterior_indices.pop();

        self.nodes_mut().map_i(root, |mut node| {
            node.radius = min_exterior_distance;
            node.interior = interior_center;
            node.exterior = exterior_center;
            node
        });

        if let Some(interior) = interior_center {
            self.nodes_mut().map_i(interior, |mut node| {
                node.parent = Some(root);
                node
            });
        }

        if let Some(exterior) = exterior_center {
            self.nodes_mut().map_i(exterior, |mut node| {
                node.parent = Some(root);
                node
            });
        }

        let mut height = 0;
        // recurse
        if let Some(x) = interior_center {
            self.bulk_build_indices(x, interior_indices);
            height = height.max(self.nodes().read(x).height);
        }

        if let Some(x) = exterior_center {
            self.bulk_build_indices(x, exterior_indices);
            height = height.max(self.nodes().read(x).height);
        }
        self.nodes_mut().map_i(root, |mut node| {
            node.height = height + 1;
            node
        });

        // #[cfg(test)]
        // {
        //     let mut post_indices = self.child_indices(root);
        //     prior_indices.sort();
        //     post_indices.sort();
        //     assert_eq!(prior_indices, post_indices);
        // }
    }

    fn set_height(&mut self, root: usize) {
        let interior_height = self
            .nodes()
            .read(root)
            .interior
            .map(|i| self.nodes().read(i).height + 1)
            .unwrap_or(0);
        let exterior_height = self
            .nodes()
            .read(root)
            .exterior
            .map(|i| self.nodes().read(i).height + 1)
            .unwrap_or(0);

        self.nodes_mut().map_i(root, |mut node| {
            node.height = interior_height.max(exterior_height);
            node
        });
    }

    fn insert_root(&mut self, root: usize, value: Self::Point) {
        // #[cfg(test)]
        // let mut prior_children = self.child_indices(root);

        let distance = self.metric().distance(
            self.node_index_data(self.nodes().read(root).center)
                .location(),
            value.location(),
        );
        let root_radius = self.nodes().read(root).radius;

        if distance < root_radius {
            // in the interior
            if let Some(ind) = self.nodes().read(root).interior {
                // recurse
                self.insert_root(ind, value);
            } else {
                let new_radius = root_radius * value.approx_halving_radius();
                // new leaf node
                self.data_mut().push(value);
                let new_index = self.n_nodes();

                self.nodes_mut().push(Node {
                    height: 0,
                    center: new_index,
                    radius: new_radius,
                    parent: Some(root),
                    interior: None,
                    exterior: None,
                });

                self.nodes_mut().map_i(root, |mut node| {
                    node.interior = Some(new_index);
                    node
                });
            }
        } else {
            if let Some(ind) = self.nodes().read(root).exterior {
                // recurse
                self.insert_root(ind, value);
            } else {
                let new_radius = root_radius * value.approx_halving_radius();

                // new leaf node
                self.data_mut().push(value);
                let new_index = self.n_nodes();

                self.nodes_mut().push(Node {
                    height: 0,
                    center: new_index,
                    radius: new_radius,
                    parent: Some(root),
                    interior: None,
                    exterior: None,
                });

                self.nodes_mut().map_i(root, |mut node| {
                    node.exterior = Some(new_index);
                    node
                });
            }
        }
        // update the height
        self.set_height(root);
        // inserted!
        // rebalance?
        // will be called again at each successively higher level

        self.rebalance(root);

        // #[cfg(test)]
        // {
        //     let mut final_children = self.child_indices(root);
        //     prior_children.push(self.n_nodes() - 1);
        //     prior_children.sort();
        //     final_children.sort();
        //
        //     assert_eq!(prior_children, final_children);
        // }
    }

    fn insert(&mut self, value: Self::Point) {
        if self.n_nodes() > 1 {
            self.insert_root(self.root(), value)
        } else if self.n_nodes() == 0 {
            self.data_mut().push(value);
            self.nodes_mut().push(Node::new_leaf(0, None))
        } else {
            let root = self.root();
            let root_dist = self.metric().distance(
                self.node_index_data(self.nodes().read(root).center)
                    .location(),
                value.location(),
            );

            self.insert_root(root, value)
        }
    }

    // insert an orphaned node
    // fn insert_existing(&mut self, root: usize, graft: usize) {
    //     let distance = self.metric().distance(
    //         self.node_index_data(self.nodes().read(root).center)
    //             .location(),
    //         self.node_index_data(graft).location(),
    //     );
    //     let root_radius = self.nodes().read(root).radius;
    //
    //     if distance < root_radius {
    //         // in the interior
    //         if let Some(ind) = self.nodes().read(root).interior {
    //             // recurse
    //             self.insert_existing(ind, graft)
    //         } else {
    //             // leaf node
    //             self.nodes_mut().map_i(root, |mut node| {
    //                 node.interior = Some(graft);
    //                 node
    //             });
    //             self.nodes_mut().map_i(graft, |mut node| {
    //                 node.radius = distance.clamp(root_radius / 2.0, root_radius);
    //                 node.parent = Some(root);
    //                 node
    //             });
    //         }
    //     } else {
    //         if let Some(ind) = self.nodes().read(root).exterior {
    //             // recurse
    //             self.insert_existing(ind, graft)
    //         } else {
    //             // leaf node
    //             self.nodes_mut().map_i(root, |mut node| {
    //                 node.exterior = Some(graft);
    //                 node
    //             });
    //             self.nodes_mut().map_i(graft, |mut node| {
    //                 node.radius = distance.clamp(root_radius / 2.0, root_radius);
    //                 node.parent = Some(root);
    //                 node
    //             });
    //         }
    //     }
    //     // update the height
    //     self.set_height(root);
    //
    //     // inserted!
    //     // rebalance?
    //     // will be called again at each successively higher level
    //     self.rebalance(self.root())
    // }

    fn rebalance(&mut self, root: usize) {
        // #[cfg(test)]
        // let mut prior_children = self.child_indices(root);

        let interior_height = self
            .nodes()
            .read(root)
            .interior
            .map(|ind| self.nodes().read(ind).height)
            .unwrap_or(0);
        let exterior_height = self
            .nodes()
            .read(root)
            .exterior
            .map(|ind| self.nodes().read(ind).height)
            .unwrap_or(0);

        if interior_height > (exterior_height + 1) {
            // interior is too big, it must be rebalanced
            self.rebalance_interior(root)
        } else if exterior_height > (interior_height + 1) {
            // exterior is too big, must be rebalanced
            self.rebalance_exterior(root)
        }

        // #[cfg(test)]
        // {
        //     let mut final_children = self.child_indices(root);
        //     prior_children.sort();
        //     final_children.sort();
        //
        //     assert_eq!(prior_children, final_children);
        // }
    }

    fn child_indices_impl(&self, root: usize, progress: &mut Vec<usize>) {
        if let Some(int) = self.nodes().read(root).interior {
            self.child_indices_impl(int, progress)
        }

        if let Some(ext) = self.nodes().read(root).exterior {
            self.child_indices_impl(ext, progress)
        }

        progress.push(root);
    }

    fn child_indices(&self, root: usize) -> Vec<usize> {
        let mut chillum = vec![];
        self.child_indices_impl(root, &mut chillum);
        chillum.pop();

        chillum
    }

    // make the interior shorter
    fn rebalance_interior(&mut self, root: usize) {
        let mut children = self.child_indices(root);

        // let root = children.pop().unwrap();
        self.bulk_build_indices(root, children);
    }

    // make the exterior shorter
    fn rebalance_exterior(&mut self, root: usize) {
        // honestly I don't see a way to be clever about this case yet.
        // rebuilding the whole dang thing
        // TODO: be good
        let mut children = self.child_indices(root);

        // let root = children.pop().unwrap();
        self.bulk_build_indices(root, children)
    }

    fn nn_iter<'a>(
        &'a self,
        query_point: &'a <Self::Point as VpTreeObject>::PointType,
    ) -> impl Iterator<Item = &'a Self::Point> {
        KnnIterator::new(query_point, self).map(|(p, _d)| p)
    }

    fn nn_dist_iter<'a>(
        &'a self,
        query_point: &'a <Self::Point as VpTreeObject>::PointType,
    ) -> KnnIterator<'a, Self> {
        KnnIterator::new(query_point, self)
    }

    fn nn_index_iter<'a>(
        &'a self,
        query_point: &'a <Self::Point as VpTreeObject>::PointType,
    ) -> KnnIndexIterator<'a, Self> {
        KnnIndexIterator::new(query_point, self)
    }

    fn check_validity_node(&self, root: usize) {
        if let Some(interior) = self.nodes().read(root).interior {
            let distance = self.metric().distance(
                self.node_index_data(root).location(),
                self.node_index_data(interior).location(),
            );

            assert!(
                distance < self.nodes().read(root).radius,
                "interior {} of {} not within radius: {} >= {}",
                interior,
                root,
                distance,
                self.nodes().read(root).radius
            );
        }

        if let Some(exterior) = self.nodes().read(root).exterior {
            let distance = self.metric().distance(
                self.node_index_data(root).location(),
                self.node_index_data(exterior).location(),
            );

            assert!(
                distance >= self.nodes().read(root).radius,
                "exterior {} of {} not outside radius: {} < {}",
                exterior,
                root,
                distance,
                self.nodes().read(root).radius
            );
        }
    }

    fn check_validity_root(&self, root: usize) {
        self.check_validity_node(root);

        if let Some(interior) = self.nodes().read(root).interior {
            self.check_validity_root(interior)
        }

        if let Some(exterior) = self.nodes().read(root).exterior {
            self.check_validity_root(exterior)
        }
    }

    fn remove(&mut self, value: &<Self::Point as VpTreeObject>::PointType) -> Option<Self::Point> {
        let mut to_remove = None;
        for (nn, _) in self
            .nn_index_iter(value)
            .take_while(|nn| nn.1 <= 0.0)
            .filter(|nn| self.data().read(nn.0).location() == value)
        {
            to_remove = Some(nn);
            break;
        }

        Some(self.remove_index(to_remove?))
    }

    // TODO: DONT BE DUM
    fn remove_index(&mut self, index: usize) -> Self::Point {
        let end = self.data_mut().pop().unwrap();
        self.nodes_mut().pop();
        let old = if index == self.n_nodes() {
            end
        } else {
            self.data_mut().replace(index, end)
        };

        let indices: Vec<usize> = (1..self.n_nodes()).collect();

        if indices.len() > 0 {
            self.bulk_build_indices(0, indices);
        } else if self.n_nodes() == 1 {
            self.nodes_mut().write(0, Node::new_leaf(0, None));
        }

        old
    }

    fn iter<'a>(&'a self) -> impl Iterator<Item = &'a Self::Point> {
        self.data().iter()
    }

    // fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Self::Point> {
    //     self.data().iter_mut()
    // }

    fn check_validity(&self) {
        if self.n_nodes() > 0 {
            self.check_validity_root(self.root())
        }
    }

    fn n_nodes(&self) -> usize {
        self.nodes().size()
    }
}

#[derive(Clone, Debug)]
struct Node {
    height: usize,
    center: usize,
    radius: f64,
    parent: Option<usize>,
    interior: Option<usize>,
    exterior: Option<usize>,
}

#[derive(Default, Debug, Clone)]
pub struct VpAvlData<
    Point,
    PointMetric,
    NodeStorage: Storage<DType = Node>,
    DataStorage: Storage<DType = Point>,
> {
    root: usize,
    nodes: NodeStorage,
    data: DataStorage,
    metric: PointMetric,
}

impl<Point, PointMetric, NodeStorage, DataStorage> VpAvl
    for VpAvlData<Point, PointMetric, NodeStorage, DataStorage>
where
    PointMetric: Metric<PointType = Point::PointType>,
    Point: VpTreeObject,
    NodeStorage: Storage<DType = Node> + Default,
    DataStorage: Storage<DType = Point> + Default,
{
    type Point = Point;
    type PointMetric = PointMetric;
    type NodeStorage = NodeStorage;
    type DataStorage = DataStorage;

    fn nodes(&self) -> &Self::NodeStorage {
        &self.nodes
    }

    fn nodes_mut(&mut self) -> &mut Self::NodeStorage {
        &mut self.nodes
    }

    fn data(&self) -> &Self::DataStorage {
        &self.data
    }

    fn data_mut(&mut self) -> &mut Self::DataStorage {
        &mut self.data
    }

    fn metric(&self) -> &Self::PointMetric {
        &self.metric
    }

    fn root(&self) -> usize {
        self.root
    }

    fn set_root(&mut self, new: usize) {
        self.root = new;
    }
}

type VpAvlVec<Point, PointMetric> = VpAvlData<Point, PointMetric, Vec<Node>, Vec<Point>>;
impl<Point, PointMetric, NodeStorage, DataStorage>
    VpAvlData<Point, PointMetric, NodeStorage, DataStorage>
where
    PointMetric: Metric<PointType = Point::PointType>,
    Point: VpTreeObject,
    NodeStorage: Storage<DType = Node> + Default,
    DataStorage: Storage<DType = Point> + Default,
{
    // fn node_index_data(&self, node_index: usize) -> &Point {
    //     &self.data().read(self.nodes().read(node_index).center)
    // }
    //
    pub fn new(metric: PointMetric) -> Self {
        VpAvlData {
            root: 0,
            nodes: Default::default(),
            data: Default::default(),
            metric,
        }
    }

    fn bulk_insert(metric: PointMetric, data: impl IntoIterator<Item = Point>) -> Self {
        let data: DataStorage = data.into_iter().collect();
        let indices: Vec<usize> = (1..data.size()).collect();
        let nodes = (0..data.size())
            .map(|ind| Node::new_leaf(ind, None))
            .collect();
        let mut rv = VpAvlData {
            root: 0,
            nodes,
            data,
            metric,
        };

        rv.bulk_build_indices(0, indices);

        rv
    }

    pub fn update_metric<NewMetric: Metric<PointType = <Point as VpTreeObject>::PointType>>(
        self,
        metric: NewMetric,
    ) -> VpAvlData<Point, NewMetric, NodeStorage, DataStorage> {
        VpAvlData::bulk_insert(metric, self.data)
    }

    //
    // pub fn update_metric<NewMetric: Metric<PointType = Point::PointType>>(
    //     self,
    //     metric: NewMetric,
    // ) -> VpAvl<Point, NewMetric, NodeStorage, DataStorage> {
    //     VpAvl::bulk_insert(metric, self.data)
    // }
    //
    // pub fn bulk_insert(metric: PointMetric, data: Vec<Point>) -> Self {
    //     let indices: Vec<usize> = (1..data.len()).collect();
    //     let nodes = (0..data.len())
    //         .map(|ind| Node::new_leaf(ind, None))
    //         .collect();
    //     let mut rv = VpAvl {
    //         root: 0,
    //         nodes,
    //         data,
    //         metric,
    //     };
    //
    //     rv.bulk_build_indices(0, indices);
    //
    //     rv
    // }
    //
    // fn bulk_build_indices(&mut self, root: usize, mut indices: Vec<usize>) {
    //     if indices.len() < 2 {
    //         // simpler case
    //         match indices.len() {
    //             0 => {
    //                 // leaf node
    //                 self.nodes().read(root).height = 0;
    //                 self.nodes().read(root).interior = None;
    //                 self.nodes().read(root).exterior = None;
    //             }
    //             1 => {
    //                 // still has one child
    //
    //                 let exterior = indices.pop().unwrap();
    //
    //                 self.nodes().read(root).exterior = Some(exterior);
    //                 self.nodes().read(root).radius = self.metric.distance(
    //                     self.node_index_data(root).location(),
    //                     self.node_index_data(exterior).location(),
    //                 );
    //                 self.nodes().read(root).height = 1;
    //                 self.nodes().read(root).interior = None;
    //                 self.nodes().read(exterior).parent = Some(root);
    //
    //                 self.bulk_build_indices(self.nodes().read(root).exterior.unwrap(), indices)
    //             }
    //             _ => unreachable!(),
    //         }
    //         return;
    //     }
    //
    //     let mut distances = Vec::with_capacity(indices.len());
    //     for index in indices.iter() {
    //         distances.push((
    //             *index,
    //             self.metric.distance(
    //                 self.node_index_data(root).location(),
    //                 self.node_index_data(*index).location(),
    //             ),
    //         ));
    //     }
    //     // sort indices by distance from root
    //     distances.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    //
    //     let partitions: Vec<&[(usize, f64)]> = distances
    //         .chunks(distances.len() / 2 + distances.len() % 2)
    //         .collect();
    //     let mut interior_indices: Vec<usize> = partitions[0].iter().map(|x| x.0).collect();
    //     let mut exterior_indices: Vec<usize> = partitions[1].iter().map(|x| x.0).collect();
    //
    //     let min_exterior_distance = partitions[1].first().unwrap().1;
    //
    //     self.nodes().read(root).radius = min_exterior_distance;
    //
    //     let interior_center = interior_indices.pop();
    //     let exterior_center = exterior_indices.pop();
    //
    //     self.nodes().read(root).interior = interior_center;
    //     self.nodes().read(root).exterior = exterior_center;
    //
    //     if let Some(interior) = interior_center {
    //         self.nodes().read(interior).parent = Some(root);
    //     }
    //
    //     if let Some(exterior) = exterior_center {
    //         self.nodes().read(exterior).parent = Some(root);
    //     }
    //
    //     let mut height = 0;
    //     // recurse
    //     if let Some(x) = interior_center {
    //         self.bulk_build_indices(x, interior_indices);
    //         height = height.max(self.nodes().read(x).height);
    //     }
    //
    //     if let Some(x) = exterior_center {
    //         self.bulk_build_indices(x, exterior_indices);
    //         height = height.max(self.nodes().read(x).height);
    //     }
    //
    //     self.nodes().read(root).height = height + 1;
    // }
    //
    // fn set_height(&mut self, root: usize) {
    //     let interior_height = self.nodes().read(root)
    //         .interior
    //         .map(|i| self.nodes().read(i).height + 1)
    //         .unwrap_or(0);
    //     let exterior_height = self.nodes().read(root)
    //         .exterior
    //         .map(|i| self.nodes().read(i).height + 1)
    //         .unwrap_or(0);
    //
    //     self.nodes().read(root).height = interior_height.max(exterior_height);
    // }
    //
    // fn insert_root(&mut self, root: usize, value: Point) {
    //     let distance = self.metric.distance(
    //         self.node_index_data(self.nodes().read(root).center).location(),
    //         value.location(),
    //     );
    //     let root_radius = self.nodes().read(root).radius;
    //
    //     if distance < root_radius {
    //         // in the interior
    //         if let Some(ind) = self.nodes().read(root).interior {
    //             // recurse
    //             self.insert_root(ind, value);
    //         } else {
    //             // new leaf node
    //             self.data.push(value);
    //             let new_index = self.data.len() - 1;
    //
    //             self.nodes.push(Node::new_leaf(new_index, Some(root)));
    //
    //             self.nodes().read(new_index).radius = distance.clamp(root_radius / 2.0, root_radius);
    //
    //             self.nodes().read(root).interior = Some(new_index);
    //         }
    //     } else {
    //         if let Some(ind) = self.nodes().read(root).exterior {
    //             // recurse
    //             self.insert_root(ind, value);
    //         } else {
    //             // new leaf node
    //             self.data.push(value);
    //             let new_index = self.data.len() - 1;
    //
    //             self.nodes.push(Node::new_leaf(new_index, Some(root)));
    //
    //             self.nodes().read(new_index).radius = distance.clamp(root_radius / 2.0, root_radius);
    //
    //             self.nodes().read(root).exterior = Some(new_index);
    //         }
    //     }
    //     // update the height
    //     self.set_height(root);
    //     // inserted!
    //     // rebalance?
    //     // will be called again at each successively higher level
    //     self.rebalance(root);
    // }
    //
    // pub fn insert(&mut self, value: Point) {
    //     if self.data.len() > 1 {
    //         self.insert_root(self.root, value)
    //     } else if self.data.len() == 0 {
    //         self.data.push(value);
    //         self.nodes.push(Node::new_leaf(0, None))
    //     } else {
    //         let root_dist = self.metric.distance(
    //             self.node_index_data(self.nodes().read(self.root).center)
    //                 .location(),
    //             value.location(),
    //         );
    //         self.nodes().read(self.root).radius = root_dist / 2.0;
    //         self.insert_root(self.root, value)
    //     }
    // }
    //
    // // insert an orphaned node
    // fn insert_existing(&mut self, root: usize, graft: usize) {
    //     let distance = self.metric.distance(
    //         self.node_index_data(self.nodes().read(root).center).location(),
    //         self.node_index_data(graft).location(),
    //     );
    //     let root_radius = self.nodes().read(root).radius;
    //
    //     if distance < root_radius {
    //         // in the interior
    //         if let Some(ind) = self.nodes().read(root).interior {
    //             // recurse
    //             self.insert_existing(ind, graft)
    //         } else {
    //             // leaf node
    //             self.nodes().read(root).interior = Some(graft);
    //             self.nodes().read(graft).radius = distance.clamp(root_radius / 2.0, root_radius);
    //             self.nodes().read(graft).parent = Some(root);
    //         }
    //     } else {
    //         if let Some(ind) = self.nodes().read(root).exterior {
    //             // recurse
    //             self.insert_existing(ind, graft)
    //         } else {
    //             // leaf node
    //             self.nodes().read(root).exterior = Some(graft);
    //             self.nodes().read(graft).radius = distance.clamp(root_radius / 2.0, root_radius);
    //             self.nodes().read(graft).parent = Some(root);
    //         }
    //     }
    //     // update the height
    //     self.set_height(root);
    //
    //     // inserted!
    //     // rebalance?
    //     // will be called again at each successively higher level
    //     self.rebalance(root)
    // }
    //
    // fn rebalance(&mut self, root: usize) {
    //     let interior_height = self.nodes().read(root)
    //         .interior
    //         .map(|ind| self.nodes().read(ind).height)
    //         .unwrap_or(0);
    //     let exterior_height = self.nodes().read(root)
    //         .exterior
    //         .map(|ind| self.nodes().read(ind).height)
    //         .unwrap_or(0);
    //
    //     if interior_height > (exterior_height + 1) {
    //         // interior is too big, it must be rebalanced
    //         self.rebalance_interior(root)
    //     } else if exterior_height > (interior_height + 1) {
    //         // exterior is too big, must be rebalanced
    //         self.rebalance_exterior(root)
    //     }
    // }
    //
    // fn child_indices_impl(&self, root: usize, progress: &mut Vec<usize>) {
    //     if let Some(int) = self.nodes().read(root).interior {
    //         self.child_indices_impl(int, progress)
    //     }
    //
    //     if let Some(ext) = self.nodes().read(root).exterior {
    //         self.child_indices_impl(ext, progress)
    //     }
    //
    //     progress.push(root);
    // }
    //
    // fn child_indices(&self, root: usize) -> Vec<usize> {
    //     let mut chillum = vec![];
    //     self.child_indices_impl(root, &mut chillum);
    //
    //     chillum
    // }
    //
    // // make the interior shorter
    // fn rebalance_interior(&mut self, root: usize) {
    //     let mut children = self.child_indices(root);
    //
    //     let root = children.pop().unwrap();
    //     self.bulk_build_indices(root, children);
    //
    //     //
    //     // The following doesn't work, because it's possible, following a bulk reindex, for the radius of a child to be larger than that of its parent.
    //     // Consequently I haven't figured out materially more efficient way of grafting the subtrees here.
    //     // I keep this in place as an inspiration to figure out how to do this properly in the future
    //     //
    //
    //     // // moves nodes as:
    //     // // interior -> root
    //     // // exterior -> new root exterior
    //     // // old root -> reinsert
    //
    //     // // there must be an interior in this case, but maybe no exterior
    //     // let old_interior = self.nodes().read(root).interior.unwrap();
    //     // let old_exterior = self.nodes().read(root).exterior;
    //
    //     // let old_root_data = self.nodes().read(root).center;
    //
    //     // println!(
    //     //     "swapping {}: {:?} <> {}: {:?}",
    //     //     root, self.nodes().read(root], old_interior, self.nodes[old_interior)
    //     // );
    //
    //     // // if there is no graft node, no children....
    //     // let mut old_exterior_children = old_exterior
    //     //     .map(|ind| self.child_indices(ind))
    //     //     .unwrap_or(vec![]);
    //
    //     // // transplant the old interior to the root
    //     // self.nodes().read(root] = self.nodes[old_interior).clone();
    //     // self.nodes().read(old_interior).center = old_root_data;
    //
    //     // let old_root_distance = self.metric.distance(
    //     //     self.node_index_data(root),
    //     //     self.node_index_data(old_interior),
    //     // );
    //
    //     // let root_radius = self.nodes().read(root).radius;
    //
    //     // // make the old root data located in the old interior node
    //     // self.nodes().read(old_interior).center = old_root_data;
    //     // self.nodes().read(old_interior).interior = None;
    //     // self.nodes().read(old_interior).exterior = None;
    //     // self.nodes().read(old_interior).height = 0;
    //     // self.nodes().read(old_interior).radius = old_root_distance.clamp(root_radius / 2.0, root_radius);
    //
    //     // // collect the new exterior nodes
    //     // let new_exterior_node = self.nodes().read(root).exterior;
    //     // // this could be empty
    //     // let mut new_exterior_children = new_exterior_node
    //     //     .map(|ind| self.child_indices(ind))
    //     //     .unwrap_or(vec![]);
    //
    //     // println!(
    //     //     "new exterior children {:?}: {:?}",
    //     //     new_exterior_node, new_exterior_children
    //     // );
    //
    //     // println!(
    //     //     "old exterior children {:?}: {:?}",
    //     //     old_exterior, old_exterior_children
    //     // );
    //
    //     // // aggregate all children...
    //     // // either or both could be empty
    //     // new_exterior_children.append(&mut old_exterior_children);
    //
    //     // // check where the old root should go...
    //     // println!(
    //     //     "swapped {}: {:?} <> {}: {:?} distance: {}/{}",
    //     //     root,
    //     //     self.nodes().read(root),
    //     //     old_interior,
    //     //     self.nodes().read(old_interior),
    //     //     old_root_distance,
    //     //     self.nodes().read(root).radius
    //     // );
    //
    //     // if old_root_distance < root_radius {
    //     //     println!(
    //     //         "old root in interior {} < {}",
    //     //         old_root_distance, root_radius
    //     //     );
    //     //     // old root is within the new root interior
    //     //     match self.nodes().read(root).interior {
    //     //         Some(interior) => self.insert_existing(interior, old_interior),
    //     //         None => {
    //     //             self.nodes().read(root).interior = Some(old_interior);
    //     //             self.nodes().read(root).radius =
    //     //                 old_root_distance.clamp(root_radius / 2.0, root_radius)
    //     //         }
    //     //     }
    //     // } else {
    //     //     println!("old root in exterior");
    //     //     // old root can be handled along with all of the other new exterior points
    //     //     new_exterior_children.push(old_interior)
    //     // }
    //
    //     // let new_exterior_root = new_exterior_children.pop();
    //     // self.nodes().read(root).exterior = new_exterior_root;
    //
    //     // // now reindex the new exterior nodes
    //     // if let Some(exterior) = new_exterior_root {
    //     //     println!(
    //     //         "new exterior nodes {}: {:?}",
    //     //         exterior, new_exterior_children
    //     //     );
    //     //     self.bulk_build_indices(exterior, new_exterior_children);
    //     // }
    //
    //     // self.set_height(root);
    //
    //     // println!(
    //     //     "finally {}: {:?} int: {:?} ext {:?}",
    //     //     root,
    //     //     self.nodes().read(root),
    //     //     self.nodes().read(root].interior.map(|i| &self.nodes[i)),
    //     //     self.nodes().read(root].exterior.map(|i| &self.nodes[i)),
    //     // );
    //
    //     // self.check_validity_root(root);
    // }
    //
    // // make the exterior shorter
    // fn rebalance_exterior(&mut self, root: usize) {
    //     // honestly I don't see a way to be clever about this case yet.
    //     // rebuilding the whole dang thing
    //     // TODO: be good
    //     let mut children = self.child_indices(root);
    //
    //     let root = children.pop().unwrap();
    //     self.bulk_build_indices(root, children)
    // }
    //
    // pub fn nn_iter<'a>(
    //     &'a self,
    //     query_point: &'a Point::PointType,
    // ) -> impl Iterator<Item = &'a Point> {
    //     KnnIterator::new(query_point, self).map(|(p, _d)| p)
    // }
    //
    // pub fn nn_dist_iter<'a>(
    //     &'a self,
    //     query_point: &'a Point::PointType,
    // ) -> KnnIterator<'a, Point, PointMetric> {
    //     KnnIterator::new(query_point, self)
    // }
    //
    // pub fn nn_iter_mut<'a>(
    //     &'a mut self,
    //     query_point: &'a Point::PointType,
    // ) -> impl Iterator<Item = &'a mut Point> {
    //     self.nn_dist_iter_mut(query_point).map(|(p, _d)| p)
    // }
    //
    // pub fn nn_dist_iter_mut<'a>(
    //     &'a mut self,
    //     query_point: &'a Point::PointType,
    // ) -> KnnMutIterator<'a, Point, PointMetric> {
    //     KnnMutIterator::new(query_point, self)
    // }
    //
    // fn nn_index_iter<'a>(
    //     &'a self,
    //     query_point: &'a Point::PointType,
    // ) -> KnnIndexIterator<'a, Point, PointMetric> {
    //     KnnIndexIterator::new(query_point, self)
    // }
    //
    // fn check_validity_node(&self, root: usize) {
    //     if let Some(interior) = self.nodes().read(root).interior {
    //         let distance = self.metric.distance(
    //             self.node_index_data(root).location(),
    //             self.node_index_data(interior).location(),
    //         );
    //
    //         assert!(
    //             distance < self.nodes().read(root).radius,
    //             "interior {} of {} not within radius: {} >= {}",
    //             interior,
    //             root,
    //             distance,
    //             self.nodes().read(root).radius
    //         );
    //     }
    //
    //     if let Some(exterior) = self.nodes().read(root).exterior {
    //         let distance = self.metric.distance(
    //             self.node_index_data(root).location(),
    //             self.node_index_data(exterior).location(),
    //         );
    //
    //         assert!(
    //             distance >= self.nodes().read(root).radius,
    //             "exterior {} of {} not outside radius: {} < {}",
    //             exterior,
    //             root,
    //             distance,
    //             self.nodes().read(root).radius
    //         );
    //     }
    // }
    //
    // fn check_validity_root(&self, root: usize) {
    //     self.check_validity_node(root);
    //
    //     if let Some(interior) = self.nodes().read(root).interior {
    //         self.check_validity_root(interior)
    //     }
    //
    //     if let Some(exterior) = self.nodes().read(root).exterior {
    //         self.check_validity_root(exterior)
    //     }
    // }
    //
    // pub fn iter<'a>(&'a self) -> impl Iterator<Item = &'a Point> {
    //     self.data.iter()
    // }
    //
    // pub fn iter_mut<'a>(&'a mut self) -> impl Iterator<Item = &'a mut Point> {
    //     self.data.iter_mut()
    // }
    //
    // fn check_validity(&self) {
    //     if self.len()() > 0 {
    //         self.check_validity_root(self.root)
    //     }
    // }
    //
    // pub fn size(&self) -> usize {
    //     self.data.len()
    // }
}

impl Node {
    fn new_leaf(center: usize, parent: Option<usize>) -> Self {
        Node {
            height: 0,
            center,
            radius: 0.0,
            interior: None,
            exterior: None,
            parent,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distributions::Uniform;
    use rand::Rng;
    use test::Bencher;

    fn check_ordering<T: VpAvl>(tree: &T, test_points: &[<T::Point as VpTreeObject>::PointType]) {
        for p in test_points.iter() {
            let mut d = 0.0;
            for (point, dist) in tree.nn_dist_iter(p) {
                assert!(dist >= d);
                d = dist;
            }
        }
    }

    #[test]
    fn test_vp() {
        let random_points = k_random(10000);
        let query_set = k_random(1000);

        let avl = VpAvlVec::bulk_insert(EuclideanMetric::default(), random_points.clone());

        assert_eq!(avl.data.len(), 10000);
        assert_eq!(avl.nodes.len(), 10000);

        // verify all nodes are children of the root
        assert_eq!(
            avl.child_indices(avl.root).len(),
            9999,
            "children: {} != 10000 - 1",
            avl.child_indices(avl.root).len()
        );

        avl.check_validity();

        let metric = EuclideanMetric::default();
        for q in query_set {
            let nn = avl.nn_iter(&q).next().unwrap();
            let avl_min_dist = metric.distance(&q, &nn);

            // linear search
            let linear_min_dist = random_points.iter().fold(f64::INFINITY, |acc, x| {
                let dist = metric.distance(&q, x);
                acc.min(dist)
            });

            assert!(
                linear_min_dist == avl_min_dist,
                "linear = {}, avl = {}",
                linear_min_dist,
                avl_min_dist
            );
        }
    }

    #[test]
    fn test_vp_incremental() {
        let random_points = k_random(10000);
        let query_set = k_random(1000);

        let metric = EuclideanMetric::default();

        let mut avl = VpAvlVec::new(metric);
        for point in random_points.iter() {
            avl.insert(point.clone());
        }

        assert!(avl.data.len() == 10000);
        assert!(avl.nodes.len() == 10000);

        // verify all nodes are children of the root
        assert_eq!(
            avl.child_indices(avl.root).len(),
            10000 - 1,
            "children: {} != 10000 - 1",
            avl.child_indices(avl.root).len()
        );

        avl.check_validity();

        let metric = EuclideanMetric::default();
        for q in query_set {
            let nn = avl.nn_iter(&q).next().unwrap();
            let avl_min_dist = metric.distance(&q, &nn);

            // linear search
            let linear_min_dist = random_points.iter().fold(f64::INFINITY, |acc, x| {
                let dist = metric.distance(&q, x);
                acc.min(dist)
            });

            assert!(
                linear_min_dist == avl_min_dist,
                "linear = {}, avl = {}",
                linear_min_dist,
                avl_min_dist
            );
        }
    }

    #[test]
    fn test_vp_remove() {
        // smaller because this is slow
        let mut random_points = k_random(1000);

        let metric = EuclideanMetric::default();

        let mut avl = VpAvlVec::new(metric);
        for point in random_points.iter() {
            avl.insert(point.clone());
        }

        assert_eq!(avl.data.len(), 1000);
        assert_eq!(avl.nodes.len(), 1000);

        // verify all nodes are children of the root
        assert_eq!(
            avl.child_indices(avl.root()).len(),
            1000 - 1,
            "children: {} != 1000 - 1",
            avl.child_indices(avl.root()).len()
        );

        avl.check_validity();

        for (ind, removal) in random_points.iter().enumerate() {
            assert!(avl.remove(&removal).is_some());
            assert!(avl.nodes.len() == (1000 - ind - 1));
            avl.check_validity();
            check_ordering(&avl, random_points.as_slice());
        }
    }

    #[test]
    fn test_reweight() {
        let random_points = k_random(10000);
        let query_set = k_random(1000);

        let avl = VpAvlVec::bulk_insert(EuclideanMetric::default(), random_points.clone());

        let weighted_metric = WeightedEuclideanMetric::new(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let new_avl = avl.update_metric(weighted_metric.clone());

        for q in query_set {
            let nn = new_avl.nn_iter(&q).next().unwrap();
            let avl_min_dist = weighted_metric.distance(&q, &nn);

            // linear search
            let linear_min_dist = random_points.iter().fold(f64::INFINITY, |acc, x| {
                let dist = weighted_metric.distance(&q, x);
                acc.min(dist)
            });

            assert_eq!(
                linear_min_dist, avl_min_dist,
                "linear = {}, avl = {}",
                linear_min_dist, avl_min_dist
            );
        }
    }

    #[test]
    fn test_iter() {
        let random_points = k_random(10000);
        let query_set = k_random(100);
        let metric = EuclideanMetric::default();
        let avl = VpAvlVec::bulk_insert(metric.clone(), random_points.clone());

        for q in query_set {
            avl.nn_iter(&q).fold(0.0, |prev, pt| {
                let dist = metric.distance(&q, pt);
                assert!(dist >= prev, "distance went down! {} < {}", dist, prev);
                dist
            });
        }
    }

    #[test]
    fn test_empty() {
        let query_set = k_random(1);
        let metric = EuclideanMetric::default();
        let avl = VpAvlVec::<Vec<f64>, _>::new(metric.clone());

        for q in query_set {
            avl.nn_iter(&q).fold(0.0, |prev, pt| {
                let dist = metric.distance(&q, pt);
                assert!(dist >= prev, "distance went down! {} < {}", dist, prev);
                dist
            });
        }
    }

    fn k_random(k: usize) -> Vec<Vec<f64>> {
        let range = Uniform::new(-1.0, 1.0);
        (0..k)
            .map(|_| rand::thread_rng().sample_iter(range).take(5).collect())
            .collect()
    }

    fn random_k(k: usize) {
        // so this is a little messy because it also generates the points, but I want to make sure the bench uses new points each time
        let points = k_random(k);
    }

    fn bench_bulk_k(k: usize) {
        // so this is a little messy because it also generates the points, but I want to make sure the bench uses new points each time
        let points = k_random(k);
        let metric = EuclideanMetric::default();
        let avl = VpAvlVec::bulk_insert(metric, points);
    }

    fn bench_incremental_k(k: usize) {
        // so this is a little messy because it also generates the points, but I want to make sure the bench uses new points each time
        let points = k_random(k);
        let metric = EuclideanMetric::default();
        let mut avl = VpAvlVec::new(metric);
        for point in points {
            avl.insert(point);
        }
    }

    #[bench]
    fn bench_random_1000(b: &mut Bencher) {
        b.iter(|| random_k(1000));
    }

    #[bench]
    fn bench_random_10000(b: &mut Bencher) {
        b.iter(|| random_k(10000));
    }

    #[bench]
    fn bench_random_100000(b: &mut Bencher) {
        b.iter(|| random_k(100000));
    }

    #[bench]
    fn bench_random_1000000(b: &mut Bencher) {
        b.iter(|| random_k(1000000));
    }

    #[bench]
    fn bench_build_vp_bulk_1000(b: &mut Bencher) {
        b.iter(|| bench_bulk_k(1000));
    }

    #[bench]
    fn bench_build_vp_incremental_1000(b: &mut Bencher) {
        b.iter(|| bench_incremental_k(1000));
    }

    #[bench]
    fn bench_build_vp_bulk_10000(b: &mut Bencher) {
        b.iter(|| bench_bulk_k(10000));
    }

    #[bench]
    fn bench_build_vp_incremental_10000(b: &mut Bencher) {
        b.iter(|| bench_incremental_k(10000));
    }

    #[bench]
    fn bench_build_vp_bulk_100000(b: &mut Bencher) {
        b.iter(|| bench_bulk_k(100000));
    }

    #[bench]
    fn bench_build_vp_incremental_100000(b: &mut Bencher) {
        b.iter(|| bench_incremental_k(100000));
    }

    // #[bench]
    // fn bench_build_vp_bulk_1000000(b: &mut Bencher) {
    //     b.iter(|| bench_bulk_k(1000000));
    // }

    // #[bench]
    // fn bench_build_vp_incremental_1000000(b: &mut Bencher) {
    //     b.iter(|| bench_incremental_k(1000000));
    // }
}
