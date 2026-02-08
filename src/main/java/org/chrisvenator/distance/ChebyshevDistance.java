package org.chrisvenator.distance;

/**
 * Chebyshev Distance metric (also known as L∞ norm or maximum metric).
 * <p>
 * The Chebyshev distance is the maximum absolute difference between coordinates.
 * It is defined as: d(a,b) = max_i |a_i - b_i|
 * </p>
 * 
 * <h2>Use Cases:</h2>
 * <ul>
 *   <li>Grid-based movement (chess king moves)</li>
 *   <li>When the maximum difference in any dimension is the limiting factor</li>
 *   <li>Warehouse logistics and robot path planning</li>
 * </ul>
 * 
 * <h2>Example:</h2>
 * <pre>
 * a = [1, 2, 3]
 * b = [4, 6, 5]
 * Chebyshev distance = max(|4-1|, |6-2|, |5-3|) = max(3, 4, 2) = 4
 * </pre>
 */
public class ChebyshevDistance implements DistanceMetric {
    
    /**
     * Calculates the Chebyshev distance between two vectors.
     *
     * @param a First vector (x, y, z, ... coordinates)
     * @param b Second vector (x, y, z, ... coordinates)
     * @return The Chebyshev distance (maximum absolute difference) from a to b
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
        
        double maxDistance = 0.0;
        
        for (int i = 0; i < a.length; i++) {
            double diff = Math.abs(a[i] - b[i]);
            if (diff > maxDistance) {
                maxDistance = diff;
            }
        }
        
        return maxDistance;
    }
    
    @Override
    public String toString() {
        return "ChebyshevDistance";
    }
}
