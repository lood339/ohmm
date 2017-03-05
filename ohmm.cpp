//
//  ohmm.cpp
//  Classifer_RF
//
//  Created by jimmy on 2017-03-04.
//  Copyright (c) 2017 Nowhere Planet. All rights reserved.
//

#include "ohmm.h"
#include <vector>

using std::vector;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixXi;



bool OHMM::ohmmOverlappingRatio(const Eigen::MatrixXd & data,
                                                double resolution,
                                                const std::vector<double> & transition,
                                                unsigned int window_size,
                                                const double overlapping_ratio,
                                                std::vector<double> & optimal_signal)
{
    assert(resolution > 0.0);
    assert(transition.size()%2 == 1);
    assert(overlapping_ratio >=0 && overlapping_ratio < 1.0);
    
   
    const double min_v = data.minCoeff();
    const double max_v = data.maxCoeff();
    const int nBin = (max_v - min_v)/resolution;
    const int moving_step = window_size * (1.0 - overlapping_ratio);  // window move forward step
    assert(moving_step >= 1);
    
    
    // raw data to probability map
    // quantilization
    const int N = (int)data.rows();
    Eigen::MatrixXd probMap = Eigen::MatrixXd::Zero(N, nBin);
    for (int r = 0; r<N; r++) {
        for (int c = 0; c<data.cols(); c++) {
            int num = valueToBinNumber(min_v, resolution, data(r, c), nBin);
            probMap(r, num) += 1.0;
        }
    }
    probMap /= data.cols(); // normalization    
    
    
    vector<double> optimalValues(N, 0);
    vector<int> numValues(N, 0);       // multiple values from local dynamic programming, as it is overlapped
    
    
    int last_index = 0;
    for (int i = 0; i <= N - window_size; i += moving_step) {
        // get a local probMap;
        MatrixXd localProbMap = probMap.middleRows(i, window_size);//probMap.extract(window_size, probMap.cols(), i, 0);
        vector<int> localOptimalBins;
        OHMM::viterbi(localProbMap, transition, localOptimalBins);
        assert(localOptimalBins.size() == window_size);
        for (int j = 0; j < localOptimalBins.size(); j++) {
            double value = OHMM::binNumberToValue(min_v, resolution, localOptimalBins[j]);
            numValues[j + i]     += 1;
            optimalValues[j + i] += value;
        }
        last_index = i;
    }
    
   
    // with fully overlapping for last several numbers
    for (int i = last_index; i <= N - window_size; i++) {
        MatrixXd localProbMap = probMap.middleRows(i, window_size); //probMap.extract(window_size, probMap.cols(), i, 0);
        vector<int> localOptimalBins;
        OHMM::viterbi(localProbMap, transition, localOptimalBins);
        assert(localOptimalBins.size() == window_size);
        for (int j = 0; j < localOptimalBins.size(); j++) {
            double value = OHMM::binNumberToValue(min_v, resolution, localOptimalBins[j]);
            numValues[j + i]     += 1;
            optimalValues[j + i] += value;
        }
    }
    assert(optimalValues.size() == N);
    
    optimal_signal.resize(N);
    // average all optimal path as final result
    for (int i = 0; i<optimalValues.size(); i++) {
        assert(numValues[i] != 0);
        optimalValues[i] /= numValues[i];
        optimal_signal[i] = optimalValues[i];
    }
    
    return true;
}

bool OHMM::viterbi(const Eigen::MatrixXd & prob_map, const vector<double> & transition,
                        vector<int> & optimal_bins)
{
    assert(transition.size()%2 == 1);
    const int N    = (int)prob_map.rows();
    const int nBin = (int)prob_map.cols();
    const int nNeighborBin = (int)transition.size()/2;
    const double epsilon = 0.01;
    
    // dynamic programming
    Eigen::MatrixXd log_accumulatedProbMap = Eigen::MatrixXd::Zero(N, nBin);
    Eigen::MatrixXi lookbackTable = Eigen::MatrixXi::Zero(N, nBin);   // store pathes
    
    // copy first row
    for (int c = 0; c<prob_map.cols(); c++) {
        log_accumulatedProbMap(0, c) = log(prob_map(0 ,c) + epsilon);
        lookbackTable(0 , c) = c;
    }
    vector<double> log_transition = vector<double>(transition.size(), 0);
    
    for (int i = 0; i<transition.size(); i++) {
        log_transition[i] = log(transition[i] + epsilon);
    }
    
    for (int r = 1; r <N; r++) {
        for (int c = 0; c<prob_map.cols(); c++) {
            // lookup all possible place in the window
            double max_val = std::numeric_limits<int>::min();
            int max_index  = -1;
            for (int w = -nNeighborBin; w <= nNeighborBin; w++) {
                if (c + w < 0 || c + w >= prob_map.cols()) {
                    continue;
                }
                assert(w + nNeighborBin >= 0 && w + nNeighborBin < transition.size());
                double val = log_accumulatedProbMap(r-1, c+w) + log_transition[w + nNeighborBin];
                if (val > max_val) {
                    max_val = val;
                    max_index = c + w; // most probable path from the [r-1] row, in column c + w
                }
            }
            assert(max_index != -1);
            log_accumulatedProbMap(r, c) = max_val + log(prob_map(r ,c) + epsilon);
            lookbackTable(r, c)          = max_index;
        }
    }
    
    // lookback the table
    double max_prob    = std::numeric_limits<int>::min();
    int max_prob_index = -1;
    for (int c = 0; c<log_accumulatedProbMap.cols(); c++) {
        if (log_accumulatedProbMap(N-1, c) > max_prob) {
            max_prob = log_accumulatedProbMap(N-1, c);
            max_prob_index = c;
        }
    }
    
    // back track
    optimal_bins.push_back(max_prob_index);
    for (int r = N-1; r > 0; r--) {
        int bin = lookbackTable(r, optimal_bins.back());
        optimal_bins.push_back(bin);
    }
    assert(optimal_bins.size() == N);
    std::reverse(optimal_bins.begin(), optimal_bins.end());
    return true;
}

unsigned OHMM::valueToBinNumber(double v_min, double interval, double value, const unsigned nBin)
{
    int num = (value - v_min)/interval;
    if (num < 0) {
        return 0;
    }
    if (num >= nBin) {
        return nBin - 1;
    }
    return (unsigned)num;
}

double OHMM::binNumberToValue(double v_min, double interval, int bin)
{
    return v_min + bin * interval;
}

