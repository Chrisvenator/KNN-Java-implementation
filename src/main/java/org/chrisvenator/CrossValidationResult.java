package org.chrisvenator;

import java.util.Arrays;

public record CrossValidationResult(double[] accuracies) {
    
    public double meanAccuracy() {
        return Arrays.stream(accuracies).average().orElse(0.0);
    }
    
    public double standardDeviation() {
        double mean = meanAccuracy();
        double variance = Arrays.stream(accuracies)
                .map(acc -> Math.pow(acc - mean, 2))
                .average()
                .orElse(0.0);
        return Math.sqrt(variance);
    }
    
    @Override
    public String toString() {
        return String.format("Cross-Validation Results:%n" +
                        "  Mean Accuracy: %.4f%n" +
                        "  Std Deviation: %.4f%n" +
                        "  Fold Accuracies: %s",
                meanAccuracy(),
                standardDeviation(),
                Arrays.toString(accuracies));
    }
}