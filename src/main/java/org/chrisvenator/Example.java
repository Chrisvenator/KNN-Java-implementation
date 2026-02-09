package org.chrisvenator;

import org.chrisvenator.Voting.MajorityVoting;
import org.chrisvenator.Voting.VotingMetric;
import org.chrisvenator.Voting.WeightedMajorityVoting;
import org.chrisvenator.distance.DistanceMetric;
import org.chrisvenator.distance.EuclideanDistance;
import org.chrisvenator.distance.ManhattanDistance;

/**
 * Demonstration of KNN classifier with various configurations.
 */
public class Example {
    static void main() {
        // Training data: two clusters representing two classes
        double[][] trainingData = {{1, 2}, {2, 3}, {3, 1},  // Class 0 (lower-left cluster)
                {6, 5}, {7, 7}, {8, 6}   // Class 1 (upper-right cluster)
        };
        int[] labels = {0, 0, 0, 1, 1, 1};
        
        // Initialize and train the classifier
        KNNClassifier knn = new KNNClassifier();
        knn.fit(trainingData, labels);
        
        System.out.println("KNN Classifier Demo");
        System.out.println("===================");
        System.out.println();
        
        // Test points
        double[] testPoint1 = {2, 2};    // Should be class 0
        double[] testPoint2 = {7, 6};    // Should be class 1
        double[] testPoint3 = {4.5, 4};  // Boundary point
        
        // Example 1: Using Euclidean distance and Majority voting
        System.out.println("Example 1: Euclidean + Majority Voting (k=3)");
        int pred1 = knn.predict(3, testPoint1, new EuclideanDistance(), new MajorityVoting());
        int pred2 = knn.predict(3, testPoint2, new EuclideanDistance(), new MajorityVoting());
        int pred3 = knn.predict(3, testPoint3, new EuclideanDistance(), new MajorityVoting());
        System.out.println("  Point " + formatPoint(testPoint1) + " -> Class " + pred1);
        System.out.println("  Point " + formatPoint(testPoint2) + " -> Class " + pred2);
        System.out.println("  Point " + formatPoint(testPoint3) + " -> Class " + pred3);
        System.out.println();
        
        // Example 2: Using Manhattan distance and Majority voting
        System.out.println("Example 2: Manhattan + Majority Voting (k=3)");
        pred1 = knn.predict(3, testPoint1, new ManhattanDistance(), new MajorityVoting());
        pred2 = knn.predict(3, testPoint2, new ManhattanDistance(), new MajorityVoting());
        pred3 = knn.predict(3, testPoint3, new ManhattanDistance(), new MajorityVoting());
        System.out.println("  Point " + formatPoint(testPoint1) + " -> Class " + pred1);
        System.out.println("  Point " + formatPoint(testPoint2) + " -> Class " + pred2);
        System.out.println("  Point " + formatPoint(testPoint3) + " -> Class " + pred3);
        System.out.println();
        
        // Example 3: Using Euclidean distance and Weighted voting
        System.out.println("Example 3: Euclidean + Weighted Voting (k=5)");
        pred1 = knn.predict(5, testPoint1, new EuclideanDistance(), new WeightedMajorityVoting());
        pred2 = knn.predict(5, testPoint2, new EuclideanDistance(), new WeightedMajorityVoting());
        pred3 = knn.predict(5, testPoint3, new EuclideanDistance(), new WeightedMajorityVoting());
        System.out.println("  Point " + formatPoint(testPoint1) + " -> Class " + pred1);
        System.out.println("  Point " + formatPoint(testPoint2) + " -> Class " + pred2);
        System.out.println("  Point " + formatPoint(testPoint3) + " -> Class " + pred3);
        System.out.println();
        
        // Example 4: Different k values comparison
        System.out.println("Example 4: Effect of different k values on boundary point " + formatPoint(testPoint3));
        for (int k = 1; k <= 6; k++) {
            int prediction = knn.predict(k, testPoint3, new EuclideanDistance(), new MajorityVoting());
            System.out.println("  k=" + k + " -> Class " + prediction);
        }
        System.out.println();
        
        // Example 5: Using defaults (null parameters)
        System.out.println("Example 5: Using defaults (null for distance and voting)");
        int defaultPred = knn.predict(3, testPoint1, null, null);
        System.out.println("  Point " + formatPoint(testPoint1) + " -> Class " + defaultPred);
        System.out.println("  (Uses EuclideanDistance and MajorityVoting by default)");
        
        
        // Example 6: Hyperparameter Search with Cross-Validation
        System.out.println("Example 6: Finding Best Hyperparameters");
        System.out.println("==========================================");
        
        HyperparameterSearchResult searchResult = knn.findBestHyperparameters(5, new int[]{1, 2, 3},
                new DistanceMetric[]{new EuclideanDistance(), new ManhattanDistance()},
                new VotingMetric[]{new MajorityVoting(), new WeightedMajorityVoting()}
        );
        
        System.out.println(searchResult);
        System.out.println();
        
        // Optional: Show detailed results
        System.out.println("Detailed Results:");
        searchResult.allResults().forEach((config, results) -> {
            System.out.println("  " + config + ":");
            results.forEach((k, cvResult) -> System.out.printf("    k=%d: %.4f (±%.4f)%n", k, cvResult.meanAccuracy(), cvResult.standardDeviation()));
        });
        
        
        
    }
    
    private static String formatPoint(double[] point) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < point.length; i++) {
            sb.append(String.format("%.1f", point[i]));
            if (i < point.length - 1) sb.append(", ");
        }
        sb.append("]");
        return sb.toString();
    }
}