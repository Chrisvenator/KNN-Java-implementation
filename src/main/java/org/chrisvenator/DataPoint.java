package org.chrisvenator;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
public class DataPoint {
    double[] vector;
    int label;
}
