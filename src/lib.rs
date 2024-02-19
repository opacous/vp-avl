#![feature(test)]
extern crate test;

use std::{collections::BinaryHeap, marker::PhantomData};

pub trait Metric {
    type PointType;
    fn distance(&self, p1: &Self::PointType, p2: &Self::PointType) -> f64;
}

pub trait VpTreeObject: Sized {
    type PointType;
    fn location(&self) -> &Self::PointType;
}

impl VpTreeObject for Vec<f64> {
    type PointType = Self;
    fn location(&self) -> &Self {
        self
    }
}

pub struct VpAvl<Point, PointMetric> {
    root: usize,
    nodes: Vec<Node>,
    data: Vec<Point>,
    metric: PointMetric,
}

#[derive(Clone, Debug)]
struct Node {
    height: usize,
    center: usize,
    radius: f64,
    interior: Option<usize>,
    exterior: Option<usize>,
}

impl<Point, PointMetric> VpAvl<Point, PointMetric>
where
    PointMetric: Metric<PointType = Point::PointType>,
    Point: VpTreeObject,
{
    fn node_index_data(&self, node_index: usize) -> &Point {
        &self.data[self.nodes[node_index].center]
    }

    pub fn new(metric: PointMetric) -> Self {
        VpAvl {
            root: 0,
            nodes: vec![],
            data: vec![],
            metric,
        }
    }

    pub fn update_metric<NewMetric: Metric<PointType = Point::PointType>>(
        self,
        metric: NewMetric,
    ) -> VpAvl<Point, NewMetric> {
        VpAvl::bulk_insert(metric, self.data)
    }

    pub fn bulk_insert(metric: PointMetric, data: Vec<Point>) -> Self {
        let indices: Vec<usize> = (1..data.len()).collect();
        let nodes = (0..data.len()).map(|ind| Node::new_leaf(ind)).collect();
        let mut rv = VpAvl {
            root: 0,
            nodes,
            data,
            metric,
        };

        rv.bulk_build_indices(0, indices);

        rv
    }

    fn bulk_build_indices(&mut self, root: usize, mut indices: Vec<usize>) {
        if indices.len() < 2 {
            // simpler case
            match indices.len() {
                0 => {
                    // leaf node
                    self.nodes[root].height = 0;
                    self.nodes[root].interior = None;
                    self.nodes[root].exterior = None;
                }
                1 => {
                    // still has one child

                    let exterior = indices.pop().unwrap();

                    self.nodes[root].exterior = Some(exterior);
                    self.nodes[root].radius = self.metric.distance(
                        self.node_index_data(root).location(),
                        self.node_index_data(exterior).location(),
                    );
                    self.nodes[root].height = 1;
                    self.nodes[root].interior = None;

                    self.bulk_build_indices(self.nodes[root].exterior.unwrap(), indices)
                }
                _ => unreachable!(),
            }
            return;
        }

        let mut distances = Vec::with_capacity(indices.len());
        for index in indices.iter() {
            distances.push((
                *index,
                self.metric.distance(
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

        self.nodes[root].radius = min_exterior_distance;

        let interior_center = interior_indices.pop();
        let exterior_center = exterior_indices.pop();

        self.nodes[root].interior = interior_center;
        self.nodes[root].exterior = exterior_center;

        let mut height = 0;
        // recurse
        if let Some(x) = interior_center {
            self.bulk_build_indices(x, interior_indices);
            height = height.max(self.nodes[x].height);
        }

        if let Some(x) = exterior_center {
            self.bulk_build_indices(x, exterior_indices);
            height = height.max(self.nodes[x].height);
        }

        self.nodes[root].height = height + 1;
    }

    fn set_height(&mut self, root: usize) {
        let interior_height = self.nodes[root]
            .interior
            .map(|i| self.nodes[i].height + 1)
            .unwrap_or(0);
        let exterior_height = self.nodes[root]
            .exterior
            .map(|i| self.nodes[i].height + 1)
            .unwrap_or(0);

        self.nodes[root].height = interior_height.max(exterior_height);
    }

    fn insert_root(&mut self, root: usize, value: Point) {
        let distance = self.metric.distance(
            self.node_index_data(self.nodes[root].center).location(),
            value.location(),
        );
        let root_radius = self.nodes[root].radius;

        if distance < root_radius {
            // in the interior
            if let Some(ind) = self.nodes[root].interior {
                // recurse
                self.insert_root(ind, value);
            } else {
                // new leaf node
                self.data.push(value);
                let new_index = self.data.len() - 1;

                self.nodes.push(Node::new_leaf(new_index));

                self.nodes[new_index].radius = distance.clamp(root_radius / 2.0, root_radius);

                self.nodes[root].interior = Some(new_index);
            }
        } else {
            if let Some(ind) = self.nodes[root].exterior {
                // recurse
                self.insert_root(ind, value);
            } else {
                // new leaf node
                self.data.push(value);
                let new_index = self.data.len() - 1;

                self.nodes.push(Node::new_leaf(new_index));

                self.nodes[new_index].radius = distance.clamp(root_radius / 2.0, root_radius);

                self.nodes[root].exterior = Some(new_index);
            }
        }
        // update the height
        self.set_height(root);
        // inserted!
        // rebalance?
        // will be called again at each successively higher level
        self.rebalance(root);
    }

    pub fn insert(&mut self, value: Point) {
        if self.data.len() > 1 {
            self.insert_root(self.root, value)
        } else if self.data.len() == 0 {
            self.data.push(value);
            self.nodes.push(Node::new_leaf(0))
        } else {
            let root_dist = self.metric.distance(
                self.node_index_data(self.nodes[self.root].center)
                    .location(),
                value.location(),
            );
            self.nodes[self.root].radius = root_dist / 2.0;
            self.insert_root(self.root, value)
        }
    }

    // insert an orphaned node
    fn insert_existing(&mut self, root: usize, graft: usize) {
        let distance = self.metric.distance(
            self.node_index_data(self.nodes[root].center).location(),
            self.node_index_data(graft).location(),
        );
        let root_radius = self.nodes[root].radius;

        if distance < root_radius {
            // in the interior
            if let Some(ind) = self.nodes[root].interior {
                // recurse
                self.insert_existing(ind, graft)
            } else {
                // leaf node
                self.nodes[root].interior = Some(graft);
                self.nodes[graft].radius = distance.clamp(root_radius / 2.0, root_radius);
            }
        } else {
            if let Some(ind) = self.nodes[root].exterior {
                // recurse
                self.insert_existing(ind, graft)
            } else {
                // leaf node
                self.nodes[root].exterior = Some(graft);
                self.nodes[graft].radius = distance.clamp(root_radius / 2.0, root_radius);
            }
        }
        // update the height
        self.set_height(root);

        // inserted!
        // rebalance?
        // will be called again at each successively higher level
        self.rebalance(root)
    }

    fn rebalance(&mut self, root: usize) {
        let interior_height = self.nodes[root]
            .interior
            .map(|ind| self.nodes[ind].height)
            .unwrap_or(0);
        let exterior_height = self.nodes[root]
            .exterior
            .map(|ind| self.nodes[ind].height)
            .unwrap_or(0);

        if interior_height > (exterior_height + 1) {
            // interior is too big, it must be rebalanced
            self.rebalance_interior(root)
        } else if exterior_height > (interior_height + 1) {
            // exterior is too big, must be rebalanced
            self.rebalance_exterior(root)
        }
    }

    fn child_indices_impl(&self, root: usize, progress: &mut Vec<usize>) {
        if let Some(int) = self.nodes[root].interior {
            self.child_indices_impl(int, progress)
        }

        if let Some(ext) = self.nodes[root].exterior {
            self.child_indices_impl(ext, progress)
        }

        progress.push(root);
    }

    fn child_indices(&self, root: usize) -> Vec<usize> {
        let mut chillum = vec![];
        self.child_indices_impl(root, &mut chillum);

        chillum
    }

    // make the interior shorter
    fn rebalance_interior(&mut self, root: usize) {
        let mut children = self.child_indices(root);

        let root = children.pop().unwrap();
        self.bulk_build_indices(root, children);

        //
        // The following doesn't work, because it's possible, following a bulk reindex, for the radius of a child to be larger than that of its parent.
        // Consequently I haven't figured out materially more efficient way of grafting the subtrees here.
        // I keep this in place as an inspiration to figure out how to do this properly in the future
        //

        // // moves nodes as:
        // // interior -> root
        // // exterior -> new root exterior
        // // old root -> reinsert

        // // there must be an interior in this case, but maybe no exterior
        // let old_interior = self.nodes[root].interior.unwrap();
        // let old_exterior = self.nodes[root].exterior;

        // let old_root_data = self.nodes[root].center;

        // println!(
        //     "swapping {}: {:?} <> {}: {:?}",
        //     root, self.nodes[root], old_interior, self.nodes[old_interior]
        // );

        // // if there is no graft node, no children....
        // let mut old_exterior_children = old_exterior
        //     .map(|ind| self.child_indices(ind))
        //     .unwrap_or(vec![]);

        // // transplant the old interior to the root
        // self.nodes[root] = self.nodes[old_interior].clone();
        // self.nodes[old_interior].center = old_root_data;

        // let old_root_distance = self.metric.distance(
        //     self.node_index_data(root),
        //     self.node_index_data(old_interior),
        // );

        // let root_radius = self.nodes[root].radius;

        // // make the old root data located in the old interior node
        // self.nodes[old_interior].center = old_root_data;
        // self.nodes[old_interior].interior = None;
        // self.nodes[old_interior].exterior = None;
        // self.nodes[old_interior].height = 0;
        // self.nodes[old_interior].radius = old_root_distance.clamp(root_radius / 2.0, root_radius);

        // // collect the new exterior nodes
        // let new_exterior_node = self.nodes[root].exterior;
        // // this could be empty
        // let mut new_exterior_children = new_exterior_node
        //     .map(|ind| self.child_indices(ind))
        //     .unwrap_or(vec![]);

        // println!(
        //     "new exterior children {:?}: {:?}",
        //     new_exterior_node, new_exterior_children
        // );

        // println!(
        //     "old exterior children {:?}: {:?}",
        //     old_exterior, old_exterior_children
        // );

        // // aggregate all children...
        // // either or both could be empty
        // new_exterior_children.append(&mut old_exterior_children);

        // // check where the old root should go...
        // println!(
        //     "swapped {}: {:?} <> {}: {:?} distance: {}/{}",
        //     root,
        //     self.nodes[root],
        //     old_interior,
        //     self.nodes[old_interior],
        //     old_root_distance,
        //     self.nodes[root].radius
        // );

        // if old_root_distance < root_radius {
        //     println!(
        //         "old root in interior {} < {}",
        //         old_root_distance, root_radius
        //     );
        //     // old root is within the new root interior
        //     match self.nodes[root].interior {
        //         Some(interior) => self.insert_existing(interior, old_interior),
        //         None => {
        //             self.nodes[root].interior = Some(old_interior);
        //             self.nodes[root].radius =
        //                 old_root_distance.clamp(root_radius / 2.0, root_radius)
        //         }
        //     }
        // } else {
        //     println!("old root in exterior");
        //     // old root can be handled along with all of the other new exterior points
        //     new_exterior_children.push(old_interior)
        // }

        // let new_exterior_root = new_exterior_children.pop();
        // self.nodes[root].exterior = new_exterior_root;

        // // now reindex the new exterior nodes
        // if let Some(exterior) = new_exterior_root {
        //     println!(
        //         "new exterior nodes {}: {:?}",
        //         exterior, new_exterior_children
        //     );
        //     self.bulk_build_indices(exterior, new_exterior_children);
        // }

        // self.set_height(root);

        // println!(
        //     "finally {}: {:?} int: {:?} ext {:?}",
        //     root,
        //     self.nodes[root],
        //     self.nodes[root].interior.map(|i| &self.nodes[i]),
        //     self.nodes[root].exterior.map(|i| &self.nodes[i]),
        // );

        // self.check_validity_root(root);
    }

    // make the exterior shorter
    fn rebalance_exterior(&mut self, root: usize) {
        // honestly I don't see a way to be clever about this case yet.
        // rebuilding the whole dang thing
        // TODO: be good
        let mut children = self.child_indices(root);

        let root = children.pop().unwrap();
        self.bulk_build_indices(root, children)
    }

    pub fn nn_iter<'a>(
        &'a self,
        query_point: &'a Point::PointType,
    ) -> impl Iterator<Item = &'a Point> {
        KnnIterator::new(query_point, self).map(|(p, _d)| p)
    }

    pub fn nn_dist_iter<'a>(
        &'a self,
        query_point: &'a Point::PointType,
    ) -> KnnIterator<'a, Point, PointMetric> {
        KnnIterator::new(query_point, self)
    }

    fn check_validity_node(&self, root: usize) {
        if let Some(interior) = self.nodes[root].interior {
            let distance = self.metric.distance(
                self.node_index_data(root).location(),
                self.node_index_data(interior).location(),
            );

            assert!(
                distance < self.nodes[root].radius,
                "interior {} of {} not within radius: {} >= {}",
                interior,
                root,
                distance,
                self.nodes[root].radius
            );
        }

        if let Some(exterior) = self.nodes[root].exterior {
            let distance = self.metric.distance(
                self.node_index_data(root).location(),
                self.node_index_data(exterior).location(),
            );

            assert!(
                distance >= self.nodes[root].radius,
                "exterior {} of {} not outside radius: {} < {}",
                exterior,
                root,
                distance,
                self.nodes[root].radius
            );
        }
    }

    fn check_validity_root(&self, root: usize) {
        self.check_validity_node(root);

        if let Some(interior) = self.nodes[root].interior {
            self.check_validity_root(interior)
        }

        if let Some(exterior) = self.nodes[root].exterior {
            self.check_validity_root(exterior)
        }
    }

    fn check_validity(&self) {
        self.check_validity_root(self.root)
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }
}

struct NodeProspect {
    index: usize,
    min_distance: f64,
}

impl PartialEq for NodeProspect {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.min_distance == other.min_distance
    }
}

impl Eq for NodeProspect {}

impl PartialOrd for NodeProspect {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // reverses comparison order to make small distances greater than large ones
        other.min_distance.partial_cmp(&self.min_distance)
    }
}

impl Ord for NodeProspect {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // reverses comparison order to make small distances greater than large ones
        other.min_distance.partial_cmp(&self.min_distance).unwrap()
    }
}

pub struct KnnIterator<'a, Point: VpTreeObject, PointMetric> {
    query_point: &'a Point::PointType,
    tree: &'a VpAvl<Point, PointMetric>,
    prospects: BinaryHeap<NodeProspect>,
    yield_queue: BinaryHeap<NodeProspect>,
}

impl<'a, Point, PointMetric> KnnIterator<'a, Point, PointMetric>
where
    Point: VpTreeObject,
    PointMetric: Metric<PointType = Point::PointType>,
{
    fn new(query_point: &'a Point::PointType, tree: &'a VpAvl<Point, PointMetric>) -> Self {
        let mut prospects = BinaryHeap::new();
        prospects.push(NodeProspect {
            index: tree.root,
            min_distance: 0.0,
        });

        KnnIterator {
            query_point,
            tree,
            prospects,
            yield_queue: BinaryHeap::new(),
        }
    }
}

impl<'a, Point, PointMetric> Iterator for KnnIterator<'a, Point, PointMetric>
where
    Point: VpTreeObject,
    PointMetric: Metric<PointType = Point::PointType>,
{
    type Item = (&'a Point, f64);

    fn next(&mut self) -> Option<Self::Item> {
        let top_choice = match self.prospects.pop() {
            Some(x) => x,
            None => {
                // nothing left to check
                return None;
            }
        };

        let center_dist = self.tree.metric.distance(
            self.query_point,
            self.tree
                .node_index_data(self.tree.nodes[top_choice.index].center)
                .location(),
        );

        // soft-yield the center
        self.yield_queue.push(NodeProspect {
            index: top_choice.index,
            min_distance: center_dist,
        });

        let diff = center_dist - self.tree.nodes[top_choice.index].radius;
        let min_interior_distance = diff.max(0.0);
        let min_exterior_distance = (-diff).max(0.0);

        if let Some(interior) = self.tree.nodes[top_choice.index].interior {
            self.prospects.push(NodeProspect {
                index: interior,
                min_distance: min_interior_distance,
            })
        }

        if let Some(exterior) = self.tree.nodes[top_choice.index].exterior {
            self.prospects.push(NodeProspect {
                index: exterior,
                min_distance: min_exterior_distance,
            })
        }

        let yield_now = if let Some(yv) = self.yield_queue.peek() {
            if let Some(pv) = self.prospects.peek() {
                if yv.min_distance <= pv.min_distance {
                    // we already have a point at least as good as any prospect
                    true
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        };

        if yield_now {
            let yv = self.yield_queue.pop().unwrap();

            Some((
                &self.tree.data[self.tree.nodes[yv.index].center],
                yv.min_distance,
            ))
        } else {
            // recurse
            self.next()
        }
    }
}

impl Node {
    fn new_leaf(center: usize) -> Self {
        Node {
            height: 0,
            center,
            radius: 0.0,
            interior: None,
            exterior: None,
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distributions::Uniform;
    use rand::Rng;
    use test::Bencher;

    #[test]
    fn test_vp() {
        let random_points = k_random(10000);
        let query_set = k_random(1000);

        let avl = VpAvl::bulk_insert(EuclideanMetric::default(), random_points.clone());

        assert!(avl.data.len() == 10000);
        assert!(avl.nodes.len() == 10000);

        // verify all nodes are children of the root
        assert!(
            avl.child_indices(avl.root).len() == 10000,
            "children: {} != 10000",
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

        let mut avl = VpAvl::new(metric);
        for point in random_points.iter() {
            avl.insert(point.clone());
        }

        assert!(avl.data.len() == 10000);
        assert!(avl.nodes.len() == 10000);

        // verify all nodes are children of the root
        assert!(
            avl.child_indices(avl.root).len() == 10000,
            "children: {} != 10000",
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
    fn test_reweight() {
        let random_points = k_random(10000);
        let query_set = k_random(1000);

        let avl = VpAvl::bulk_insert(EuclideanMetric::default(), random_points.clone());

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

            assert!(
                linear_min_dist == avl_min_dist,
                "linear = {}, avl = {}",
                linear_min_dist,
                avl_min_dist
            );
        }
    }

    #[test]
    fn test_iter() {
        let random_points = k_random(10000);
        let query_set = k_random(100);
        let metric = EuclideanMetric::default();
        let avl = VpAvl::bulk_insert(metric.clone(), random_points.clone());

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
        let avl = VpAvl::bulk_insert(metric, points);
    }

    fn bench_incremental_k(k: usize) {
        // so this is a little messy because it also generates the points, but I want to make sure the bench uses new points each time
        let points = k_random(k);
        let metric = EuclideanMetric::default();
        let mut avl = VpAvl::new(metric);
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
