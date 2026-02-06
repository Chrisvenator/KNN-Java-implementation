package org.chrisvenator.distance;

import java.util.stream.IntStream;

public class ManhattanDistance implements DistanceMetric {
    @Override
    public double calculateDistance(double[] a, double[] b) {
        if (a == null || b == null) throw new IllegalArgumentException("a must not be null!");
        if (a.length != b.length) throw new IllegalArgumentException("a and b length are different!");
        if (a.length == 0) return 0;
        
        return IntStream.range(0, a.length)
                .asDoubleStream()
                .map(i -> Math.abs(a[(int) i] - b[(int) i]))
                .sum();
    }
}
