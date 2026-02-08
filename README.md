# K-Nearest Neighbors (KNN) Classifier - Java Implementation

A flexible, well-tested Java implementation of the K-Nearest Neighbors classification algorithm with support for multiple distance metrics and voting strategies.

## Features

**Distance Metrics:**
- Euclidean Distance (L2 norm)
- Manhattan Distance (L1 norm)
- Minkowski Distance (generalized Lp norm)
- Chebyshev Distance (L∞ norm)

**Voting Strategies:**
- Majority Voting (simple majority)
- Weighted Voting (inverse distance weighting)


## Quick Start

### Basic Usage

```java
import org.chrisvenator.KNNClassifier;
import org.chrisvenator.Voting.MajorityVoting;
import org.chrisvenator.distance.EuclideanDistance;

// 1. Prepare training data
double[][] trainingData = {{1, 2}, {2, 3}, {3, 1}, {6, 5}, {7, 7}, {8, 6}};
int[] labels = {0, 0, 0, 1, 1, 1};

// 2. Create and train classifier
KNNClassifier knn = new KNNClassifier();
knn.fit(trainingData, labels);

// 3. Make predictions
double[] testPoint = {2, 2};
int prediction = knn.predict(3, testPoint, new EuclideanDistance(), new MajorityVoting());

System.out.println("Predicted class: " + prediction);
```

### Using Different Distance Metrics

```java
// Manhattan Distance
int pred1 = knn.predict(3, testPoint, new ManhattanDistance(), new MajorityVoting());

// Minkowski Distance (p=3)
int pred2 = knn.predict(3, testPoint, new MinkowskiDistance(3), new MajorityVoting());

// Chebyshev Distance
int pred3 = knn.predict(3, testPoint, new ChebyshevDistance(), new MajorityVoting());
```

### Using Weighted Voting

```java
// Weighted voting gives more importance to closer neighbors
int prediction = knn.predict(5, testPoint, 
    new EuclideanDistance(), 
    new WeightedMajorityVoting());
```

### Convenience Method

```java
// Use defaults: Euclidean distance + Majority voting
int prediction = knn.predict(3, testPoint);
```

## Distance Metrics

### Euclidean Distance
The straight-line distance between two points.
```
d(a,b) = √(Σ(ai - bi)²)
```
**Use when:** Features have similar scales, geometric distance is meaningful

### Manhattan Distance
Sum of absolute differences (taxicab distance).
```
d(a,b) = Σ|ai - bi|
```
**Use when:** Features represent independent dimensions, grid-like structure

### Minkowski Distance
Generalization of Euclidean and Manhattan distances.
```
d(a,b) = (Σ|ai - bi|^p)^(1/p)
```
- p=1: Manhattan distance
- p=2: Euclidean distance
- p→∞: Chebyshev distance

**Use when:** You want to tune the distance metric

### Chebyshev Distance
Maximum coordinate difference.
```
d(a,b) = max|ai - bi|
```
**Use when:** The limiting factor is the maximum difference in any dimension (e.g., chess king moves)

## API Reference

### KNNClassifier

#### Constructor
```java
KNNClassifier()
```

#### Methods

```java
void fit(double[][] trainingData, int[] labels)
```
Train the classifier with data.
- **Parameters:**
    - `trainingData`: 2D array where each row is a sample
    - `labels`: Array of integer labels for each sample
- **Throws:** `IllegalArgumentException` if inputs are invalid

```java
int predict(int k, double[] testPoint, DistanceMetric metric, VotingMetric voting)
```
Predict the class label for a test point.
- **Parameters:**
    - `k`: Number of nearest neighbors (must be > 0 and ≤ training size)
    - `testPoint`: Point to classify
    - `metric`: Distance metric to use (null = Euclidean)
    - `voting`: Voting strategy (null = Majority)
- **Returns:** Predicted class label
- **Throws:** `IllegalArgumentException` if inputs are invalid

```java
int predict(int k, double[] testPoint)
```
Convenience method using default metrics (Euclidean + Majority).

## Building and Testing

### Build
```bash
mvn clean compile
```

### Run Tests
```bash
mvn test
```

### Run Example
```bash
mvn exec:java -Dexec.mainClass="org.chrisvenator.App"
```

## Project Structure

```
src/
├── main/java/org/chrisvenator/
│   ├── App.java                          # Example usage
│   ├── KNNClassifier.java                # Main classifier
│   ├── DataPoint.java                    # Data point representation
│   ├── distance/
│   │   ├── DistanceMetric.java           # Distance metric interface
│   │   ├── EuclideanDistance.java
│   │   ├── ManhattanDistance.java
│   │   ├── MinkowskiDistance.java
│   │   └── ChebyshevDistance.java
│   └── Voting/
│       ├── VotingMetric.java             # Voting strategy interface
│       ├── MajorityVoting.java
│       └── WeightedMajorityVoting.java
└── test/java/org/chrisvenator/
    ├── KNNClassifierTest.java            # Comprehensive KNN tests
    ├── VotingMetricTest.java             # Voting strategy tests
    └── distance/
        ├── EuclideanDistanceTest.java
        ├── ManhattanDistanceTest.java
        └── DistanceMetricTests.java      # Minkowski & Chebyshev tests
```

## Requirements

- Java 17+
- Maven 3.6+
- JUnit 5 (for testing)
- Lombok (for boilerplate reduction)

## Advanced Topics

### Choosing k
- **Small k (1-3):** More sensitive to noise, flexible decision boundaries
- **Large k (7-15):** Smoother decision boundaries, more robust to noise
- **Rule of thumb:** k = √n where n = number of training samples
- **Best practice:** Use cross-validation to find optimal k

### Choosing Distance Metric
- **Euclidean:** Default choice for most problems
- **Manhattan:** When features are independent or on different scales
- **Minkowski (p>2):** To reduce influence of large differences
- **Chebyshev:** When the maximum difference is the limiting factor

### Performance Considerations
- **Time Complexity:** O(n·d) per prediction where n = training samples, d = dimensions
- **Space Complexity:** O(n·d) for storing training data
- **Optimization:** For large datasets, consider using KD-trees or approximate nearest neighbors

## Contributing

Suggestions for improvement:
1. Add cross-validation support
2. Add evaluation metrics (accuracy, precision, recall, F1)
3. Add batch prediction support
4. Add feature scaling/normalization
5. Implement KD-tree for faster neighbor search
6. Add model serialization (save/load)

## License

This project is available for educational and personal use.

## Author

Implementation by org.chrisvenator

## Acknowledgments

- Based on the K-Nearest Neighbors algorithm by Fix and Hodges (1951)
- Distance metrics based on Minkowski's work
- Implementation inspired by scikit-learn's KNeighborsClassifier