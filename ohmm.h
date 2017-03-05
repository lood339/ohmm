//
//  ohmm.h
//  Classifer_RF
//
//  Created by jimmy on 2017-03-04.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#ifndef __Classifer_RF__ohmm__
#define __Classifer_RF__ohmm__

// overlapped hidden Markov model
// Where should cameras look at soccer games: improving smoothness using the overlapped hidden Markov model. CVIU 2016
#include <stdio.h>
#include <Eigen/Dense>
#include <vector>




class OHMM
{
public:
    // overlapped hidden Markov model viterbi algorithm
    // data: each row is the prediction from multiple trees for a time instance
    // resolution: camera angle resolution, continous -- > discrete, default: 0.1 seconds
    // transition: transition probability. Its resolution should be the same as in "resolution". from -2 1 0 1 2
    // windowSize: sliding window size
    // overlapping_ratio: 0 - w-1/w, like: 0, 0.2, ... 0.8
    // optimal_signal: output
    static bool ohmmOverlappingRatio(const Eigen::MatrixXd & data,
                                     double resolution,
                                     const std::vector<double> & transition,
                                     unsigned int window_size,
                                     const double overlapping_ratio,
                                     std::vector<double> & optimal_signal);
    
private:
    // quantilization method
    // interval: resolution, the width of bin
    // nBin: tobal number of bins
    static unsigned valueToBinNumber(double v_min, double interval, double value, const unsigned nBin);
    
    static double binNumberToValue(double v_min, double interval, int bin);
    
    // prob_map: confusion matrix, p(x | state)
    // transition: state trainsition vector (one dimension)
    // optimal_bins: bin numbers from backtrack
    static bool viterbi(const Eigen::MatrixXd & prob_map,
                        const std::vector<double> & transition,
                        std::vector<int> & optimal_bins);
    
    
};


#endif /* defined(__Classifer_RF__ohmm__) */
