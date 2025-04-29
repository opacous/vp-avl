use crate::{Metric, Storage, VpAvl, VpTreeObject};
use std::collections::BinaryHeap;

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

pub struct KnnIterator<'a, VpTree: VpAvl> {
    query_point: &'a <VpTree::Point as VpTreeObject>::PointType,
    tree: &'a VpTree,
    prospects: BinaryHeap<NodeProspect>,
    yield_queue: BinaryHeap<NodeProspect>,
}

impl<'a, VpTree: VpAvl> KnnIterator<'a, VpTree> {
    pub fn new(
        query_point: &'a <VpTree::Point as VpTreeObject>::PointType,
        tree: &'a VpTree,
    ) -> Self {
        let mut prospects = BinaryHeap::new();
        if tree.n_nodes() > 0 {
            prospects.push(NodeProspect {
                index: tree.root(),
                min_distance: 0.0,
            });
        }

        KnnIterator {
            query_point,
            tree,
            prospects,
            yield_queue: BinaryHeap::new(),
        }
    }
}

impl<'a, VpTree: VpAvl> Iterator for KnnIterator<'a, VpTree> {
    type Item = (&'a VpTree::Point, f64);

    fn next(&mut self) -> Option<Self::Item> {
        let top_choice = match self.prospects.pop() {
            Some(x) => x,
            None => {
                // nothing left to check
                return self
                    .yield_queue
                    .pop()
                    .map(|p| (self.tree.data().read(p.index), p.min_distance));
            }
        };

        let center_dist = self.tree.metric().distance(
            self.query_point,
            self.tree
                .node_index_data(self.tree.nodes().read(top_choice.index).center)
                .location(),
        );

        // soft-yield the center
        self.yield_queue.push(NodeProspect {
            index: top_choice.index,
            min_distance: center_dist,
        });

        let diff = center_dist - self.tree.nodes().read(top_choice.index).radius;
        let min_interior_distance = diff.max(0.0);
        let min_exterior_distance = (-diff).max(0.0);

        if let Some(interior) = self.tree.nodes().read(top_choice.index).interior {
            self.prospects.push(NodeProspect {
                index: interior,
                min_distance: min_interior_distance,
            })
        }

        if let Some(exterior) = self.tree.nodes().read(top_choice.index).exterior {
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
                &self
                    .tree
                    .data()
                    .read(self.tree.nodes().read(yv.index).center),
                yv.min_distance,
            ))
        } else {
            // recurse
            self.next()
        }
    }
}

// pub struct KnnMutIterator<'a, VpTree: VpAvl> {
//     query_point: &'a <VpTree::Point as VpTreeObject>::PointType,
//     tree: &'a mut VpTree,
//     prospects: BinaryHeap<NodeProspect>,
//     yield_queue: BinaryHeap<NodeProspect>,
// }
//
// impl<'a, VpTree: VpAvl> KnnMutIterator<'a, VpTree> {
//     fn new(
//         query_point: &'a <VpTree::Point as VpTreeObject>::PointType,
//         tree: &'a mut VpTree,
//     ) -> Self {
//         let mut prospects = BinaryHeap::new();
//         if tree.size() > 0 {
//             prospects.push(NodeProspect {
//                 index: tree.root,
//                 min_distance: 0.0,
//             });
//         }
//
//         KnnMutIterator {
//             query_point,
//             tree,
//             prospects,
//             yield_queue: BinaryHeap::new(),
//         }
//     }
// }
//
// impl<'a, VpTree: VpAvl> Iterator for KnnMutIterator<'a, VpTree> {
//     type Item = (&'a mut <VpTree::Point as VpTreeObject>::PointType, f64);
//
//     fn next(&mut self) -> Option<Self::Item> {
//         let top_choice = match self.prospects.pop() {
//             Some(x) => x,
//             None => {
//                 // nothing left to check
//                 return self.yield_queue.pop().map(|p| {
//                     let target_loc: *mut <VpTree::Point as VpTreeObject>::PointType =
//                         &mut self.tree.data[p.index]
//                             as *mut <VpTree::Point as VpTreeObject>::PointType;
//                     let rv: &'a mut <VpTree::Point as VpTreeObject>::PointType =
//                         unsafe { &mut *target_loc };
//
//                     (rv, p.min_distance)
//                 });
//             }
//         };
//
//         let center_dist = self.tree.metric.distance(
//             self.query_point,
//             self.tree
//                 .node_index_data(self.tree.nodes[top_choice.index].center)
//                 .location(),
//         );
//
//         // soft-yield the center
//         self.yield_queue.push(NodeProspect {
//             index: top_choice.index,
//             min_distance: center_dist,
//         });
//
//         let diff = center_dist - self.tree.nodes[top_choice.index].radius;
//         let min_interior_distance = diff.max(0.0);
//         let min_exterior_distance = (-diff).max(0.0);
//
//         if let Some(interior) = self.tree.nodes[top_choice.index].interior {
//             self.prospects.push(NodeProspect {
//                 index: interior,
//                 min_distance: min_interior_distance,
//             })
//         }
//
//         if let Some(exterior) = self.tree.nodes[top_choice.index].exterior {
//             self.prospects.push(NodeProspect {
//                 index: exterior,
//                 min_distance: min_exterior_distance,
//             })
//         }
//
//         let yield_now = if let Some(yv) = self.yield_queue.peek() {
//             if let Some(pv) = self.prospects.peek() {
//                 if yv.min_distance <= pv.min_distance {
//                     // we already have a point at least as good as any prospect
//                     true
//                 } else {
//                     false
//                 }
//             } else {
//                 false
//             }
//         } else {
//             false
//         };
//
//         if yield_now {
//             let yv = self.yield_queue.pop().unwrap();
//
//             let target_loc: *mut <VpTree::Point as VpTreeObject>::PointType =
//                 &mut self.tree.data[yv.index] as *mut <VpTree::Point as VpTreeObject>::PointType;
//             let rv: &'a mut <VpTree::Point as VpTreeObject>::PointType =
//                 unsafe { &mut *target_loc };
//
//             Some((rv, yv.min_distance))
//         } else {
//             // recurse
//             self.next()
//         }
//     }
// }

pub struct KnnIndexIterator<'a, VpTree: VpAvl> {
    query_point: &'a <VpTree::Point as VpTreeObject>::PointType,
    tree: &'a VpTree,
    prospects: BinaryHeap<NodeProspect>,
    yield_queue: BinaryHeap<NodeProspect>,
}

impl<'a, VpTree: VpAvl> KnnIndexIterator<'a, VpTree> {
    pub fn new(
        query_point: &'a <VpTree::Point as VpTreeObject>::PointType,
        tree: &'a VpTree,
    ) -> Self {
        let mut prospects = BinaryHeap::new();
        if tree.n_nodes() > 0 {
            prospects.push(NodeProspect {
                index: tree.root(),
                min_distance: 0.0,
            });
        }

        KnnIndexIterator {
            query_point,
            tree,
            prospects,
            yield_queue: BinaryHeap::new(),
        }
    }
}

impl<'a, VpTree: VpAvl> Iterator for KnnIndexIterator<'a, VpTree> {
    type Item = (usize, f64);

    fn next(&mut self) -> Option<Self::Item> {
        let top_choice = match self.prospects.pop() {
            Some(x) => x,
            None => {
                // println!("no prospect yq: {}", self.yield_queue.len());

                return self.yield_queue.pop().map(|p| (p.index, p.min_distance));
                // nothing left to check
                // return None;
            }
        };

        let center_dist = self.tree.metric().distance(
            self.query_point,
            self.tree
                .node_index_data(self.tree.nodes().read(top_choice.index).center)
                .location(),
        );

        // soft-yield the center
        self.yield_queue.push(NodeProspect {
            index: top_choice.index,
            min_distance: center_dist,
        });

        let diff = center_dist - self.tree.nodes().read(top_choice.index).radius;
        let min_interior_distance = diff.max(0.0);
        let min_exterior_distance = (-diff).max(0.0);

        if let Some(interior) = self.tree.nodes().read(top_choice.index).interior {
            self.prospects.push(NodeProspect {
                index: interior,
                min_distance: min_interior_distance,
            })
        }

        if let Some(exterior) = self.tree.nodes().read(top_choice.index).exterior {
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
            // println!("yield: {} {} ", yv.index,
            //          yv.min_distance);
            Some((yv.index, yv.min_distance))
        } else {
            // recurse
            self.next()
        }
    }
}
