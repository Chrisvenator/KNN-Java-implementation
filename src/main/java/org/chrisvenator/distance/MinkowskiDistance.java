package org.chrisvenator.distance;

import lombok.Getter;

/**
 * Minkowski Distance metric.
 * <p>
 * The Minkowski distance is a generalization of both Euclidean and Manhattan distances.
 * It is defined as: d(a,b) = (Σ|a_i - b_i|^p)^(1/p)
 * </p>
 *
 * <h2>Special Cases:</h2>
 * <ul>
 *   <li>p = 1: Manhattan distance (L1 norm)</li>
 *   <li>p = 2: Euclidean distance (L2 norm)</li>
 *   <li>p → ∞: Chebyshev distance (L∞ norm)</li>
 * </ul>
 *
 * <h2>Usage:</h2>
 * <pre>
 * // Euclidean distance
 * DistanceMetric metric = new MinkowskiDistance(2);
 *
 * // Manhattan distance
 * DistanceMetric metric = new MinkowskiDistance(1);
 *
 * // Higher order Minkowski
 * DistanceMetric metric = new MinkowskiDistance(3);
 * </pre>
 */
@Getter
public class MinkowskiDistance implements DistanceMetric {
    
    private final double p;
    
    /**
     * Constructs a Minkowski distance metric with the specified order.
     *
     * @param p The order parameter (must be >= 1)
     * @throws IllegalArgumentException if p < 1
     */
    public MinkowskiDistance(double p) {
        if (p < 1.0) {
            throw new IllegalArgumentException(
                    String.format("p must be >= 1, but got p=%.2f", p)
            );
        }
        this.p = p;
    }
    
    /**
     * Calculates the Minkowski distance between two vectors.
     *
     * @param a First vector (x, y, z, ... coordinates)
     * @param b Second vector (x, y, z, ... coordinates)
     * @return The Minkowski distance from a to b
     * @throws IllegalArgumentException if a or b is null, or if they have different lengths
     */
    @Override
    public double calculateDistance(double[] a, double[] b) {
        if (a == null || b == null) {
            throw new IllegalArgumentException("Vectors must not be null!");
        }
        if (a.length != b.length) {
            throw new IllegalArgumentException(
                    String.format("Vectors must have same length! a.length=%d, b.length=%d",
                            a.length, b.length)
            );
        }
        if (a.length == 0) {
            return 0.0;
        }
        
        double sum = 0.0;
        
        // Special case: p = 1 (Manhattan)
        if (p == 1.0) {
            for (int i = 0; i < a.length; i++) {
                sum += Math.abs(a[i] - b[i]);
            }
            return sum;
        }
        
        // Special case: p = 2 (Euclidean)
        if (p == 2.0) {
            for (int i = 0; i < a.length; i++) {
                double diff = a[i] - b[i];
                sum += diff * diff;
            }
            return Math.sqrt(sum);
        }
        
        // General case
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(Math.abs(a[i] - b[i]), p);
        }
        
        return Math.pow(sum, 1.0 / p);
    }
    
    @Override
    public String toString() {
        return String.format("MinkowskiDistance(p=%.2f)", p);
    }
}