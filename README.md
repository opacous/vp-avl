# VP-AVL
This provides a novel type of VP-tree that also integrates features of an AVL tree, which allows insertion into the tree without complete rebuilding.
In practice building the tree in bulk is still faster (empirically by about 40% for a tree with 1000 elements up to about a factor of 2.5 for a tree of 1,000,000), but obviously still much faster than rebuilding the tree upon every insertion.

To bulk insert with the provided Euclidean Metric and some type which implements `IntoIterator`, simply:
```
        let random_points = k_random(10000);

        let avl = VpAvl::bulk_insert(EuclideanMetric::default(), random_points);

```

or iteratively:
```
        let random_points = k_random(10000);

        let avl = VpAvl::new(EuclideanMetric::default());

        for point in random_points{
            avl.insert(point)
        }
```


To iterate through the nearest neighbors to a points, call `nn_iter`:
```
    let query_point = vec![1.0,1.0,2.0,3.0,5.0];

    for point in avl.nn_iter(&query_point){
        ...
    }
```
