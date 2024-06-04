#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

#define FILEPATH "/Users/aadilzikre/Documents/Personal/notebooks/Rough/housing_cpp.csv"

time_t now = time(nullptr) ;

#define time_now put_time(gmtime(&now), "%F, %T :: ")

VectorXi getMinColumnIndexPerRow(const MatrixXd& matrix) {
    if (matrix.size() == 0) {
        // Handle the case when the matrix is empty
        return VectorXi();
    }
    VectorXi minColIndices(matrix.rows());
    for (int i = 0; i < matrix.rows(); ++i) {
        minColIndices(i) = matrix.row(i).minCoeff();
    }
    return minColIndices;
}

int main() {
    clog << time_now << "Start of Program!" << endl;
    
    ifstream iFile;
    string line = "";
    iFile.open(FILEPATH);
    int TOTAL_ROWS = 0;
    int TOTAL_COLUMNS = 0;
    while (getline(iFile, line)) {
        TOTAL_ROWS++;
        if (TOTAL_ROWS == 1) {
            int loc_comma = line.find(",");
            clog << time_now << "Columns :: " << endl; 
            while (loc_comma > 0) {
                cout << line.substr(0, loc_comma) << endl; 
                TOTAL_COLUMNS++;
                line = line.substr(loc_comma + 2, line.length() + 1);
                loc_comma = line.find(",");
                }
            TOTAL_COLUMNS++; // If we do not find a comma at the end, 
                             // that means there was atleast 1 more column
        }
    }
    cout << time_now << "total rows :: " << TOTAL_ROWS << endl;
    cout << time_now << "total columns :: " << TOTAL_COLUMNS << endl;
    iFile.close();

    MatrixXd X(TOTAL_ROWS-1, TOTAL_COLUMNS);

    iFile.open(FILEPATH);
    for (int r = 0 ; r < TOTAL_ROWS ; r++) {   
        if (r == 0) {
            getline(iFile, line);
            continue;
        } //header 
        int c = 0;
        getline(iFile, line);
        int loc_comma = line.find(",");
        // cout << loc_comma << " : "<< line.substr(0, loc_comma) << " : " << line.substr(loc_comma + 2, string::npos) << endl;
        X(r-1,c) = stod(line.substr(0, loc_comma-1));
        line = line.substr(loc_comma + 2, string::npos);
        while (loc_comma > 0) {
            c++;
            loc_comma = line.find(",");
            X(r-1,c) = stod(line.substr(0, loc_comma));
            // cout << loc_comma << " : "<< line.substr(0, loc_comma+1) << " : " << line.substr(loc_comma + 2, string::npos) << endl;
            line = line.substr(loc_comma + 1, string::npos);
            }
    }
    iFile.close();

    clog << time_now << "First Five Rows" << endl; 
    clog << "-------------" << endl; 
    clog << X(seq(0,4), all) << endl;
    clog << time_now << "Last Five Rows" << endl; 
    clog << "-------------" << endl; 
    clog << X(seq(TOTAL_ROWS-6, TOTAL_ROWS -2), all) << endl;

    int K = 4;
    
    MatrixXd centroids = MatrixXd::Random(K, TOTAL_COLUMNS);
    VectorXi labels(TOTAL_ROWS - 1);
    MatrixXd dist_centroids(TOTAL_ROWS-1, K);

    double mean_centroids = centroids.mean();
    cout << mean_centroids << endl;

    MatrixXd centroid_sq_diffs;
    centroid_sq_diffs = (centroids.array() - mean_centroids).square();

    double std = sqrt(centroid_sq_diffs.mean());

    centroids = (centroids.array() - mean_centroids) / std;

    clog << time_now << "Centroids " << endl;
    clog << time_now << " --------------- " << endl;
    clog << centroids;

    VectorXd meanX = X.colwise().mean();
    MatrixXd X_sq_diffs;
    X_sq_diffs = (X.rowwise() - meanX.transpose()).array().square();
    VectorXd stdX = sqrt(X_sq_diffs.colwise().mean().array());

    X = X.rowwise() - meanX.transpose();
    X = X.array().rowwise() / stdX.array().transpose();

    clog << time_now << "First Five Rows" << endl; 
    clog << "-------------" << endl; 
    clog << X(seq(0,4), all) << endl;
    
    const int MAX_TRAINING_STEPS = 100;
    
    MatrixXd prev_centroids = centroids;

    for (int step=0; step < MAX_TRAINING_STEPS; step++) {
        if (step%10==0) clog << time_now << step << " epochs done!" << endl;
        for (int k=0; k<K; k++) {
            dist_centroids(all, k) = (X.rowwise() - centroids(k, all)).array().square().rowwise().sum();
            labels = getMinColumnIndexPerRow(dist_centroids);
        }
        for (int k=0; k<K; k++) {
            labels = getMinColumnIndexPerRow(dist_centroids);
            Array<bool, Dynamic, 1> boolVec = (labels.array() == k);
            centroids(k,all) = X(boolVec, all).colwise().mean();
        }
        if (step%50==0) {
            clog << time_now << "Centroids at " << step << endl;
            clog << "----------------" << endl;
            clog << centroids << endl;
        }
        double centroid_diff =(prev_centroids - centroids).sum();
        // clog << time_now << "Centroid Diff at " << step << " = " << centroid_diff << endl;
        if ( centroid_diff == 0) {
            clog << time_now << "Training Converged at " << step << "!!!!" << endl;   
            break;
        } else {
            prev_centroids = centroids;
        }
    }
    clog << time_now << "Final Centroids " << endl;
    clog << "----------------" << endl;
    clog << centroids << endl;
    return 0;
}