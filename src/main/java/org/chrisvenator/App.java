package org.chrisvenator;

/**
 * Hello world!
 */
public class App {
    public static void main(String[] args) {
        double[][] trainingData = {{1, 2}, {2, 3}, {3, 1}, {6, 5}, {7, 7}, {8, 6}};
        int[] labels = {0, 0, 0, 1, 1, 1};
        
        KNNClassifier knn = new KNNClassifier(1);
        knn.fit(trainingData, labels);
        
        double[] test = {2, 2};
        knn.predict(test);
    }
}
