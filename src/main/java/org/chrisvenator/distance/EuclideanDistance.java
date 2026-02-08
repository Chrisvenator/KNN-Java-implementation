package org.chrisvenator.distance;

public class EuclideanDistance implements DistanceMetric {
    /**
     * Calculates the Euclidean distance from a to b
     *
     * @param a x, y, z, ... coordinates of first vector
     * @param b x, y, z, ... coordinates of second vector
     * @return distance from a to b
     */
    @Override
    public double calculateDistance(double[] a, double[] b) {
        if (a == null || b == null) throw new IllegalArgumentException("a must not be null!");
        if (a.length != b.length) throw new IllegalArgumentException("a and b length are different!");
        if (a.length == 0) return 0;
        
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += Math.pow(b[i] - a[i], 2);
        }
        return Math.sqrt(sum);
    }
    
    @Override
    public String toString() {
        return "EuclideanDistance";
    }
}