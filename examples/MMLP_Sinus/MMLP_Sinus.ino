/*
  Test Sinus with optimize (automatic heuristics)
  (c) Lesept - January 2022

*/
#include <Matrix.h>
#include <MMLP.h>
#define FORMAT_LITTLEFS_IF_FAILED true
const char networkFile[] = "/Network.txt";

void setup ()
{
  Serial.begin (115200);
  delay(200);
  if (!LITTLEFS.begin(FORMAT_LITTLEFS_IF_FAILED)) {
    Serial.println("LITTLEFS Mount Failed");
    return;
  }

  float eta = 0.5f; // learning rate
  float momentum = 0.1f;
  int Neurons[] = {1, 8, 3, 1};   // neurons per layer
  int activation[] = {TANH, TANH, TANH}; // activations of each layer
  int maxEpochs = 300;
  float stopError = 0.02f;
  int batchSize = 2;

  MLP Net (Neurons, 4, 1);
  Net.setActivations(activation);
  Net.setHyper(eta, momentum);
  // Display size of network
  Net.size();

  // Create the dataset
  //////////////    SINUS    //////////////

  int nData = 150;
  //  float x[nData], y[nData];
  std::vector<float> x, y;
  for (int i = 0; i < nData; ++i) {
    float xx = -PI + 2.0f * PI * float(i) / (nData - 1.0f);
    float yy = sin(xx);
    //    x[i] = xx;
    //    y[i] = yy;
    x.push_back(xx);
    y.push_back(yy);
  }

  ///////////////////////////////////////////
  
  MLMatrix<float> dataX;
  MLMatrix<float> dataY;
  //  Net.createDatasetFromArray (dataX, dataY, x, y, nData);
  Net.createDatasetFromVector (dataX, dataY, x, y);
  Net.setTrainTest(4, 1, 1);
 // Net.normalizeDataset(dataX, dataY, 2);
  Net.run (dataX, dataY, maxEpochs, batchSize, stopError);
  Net.netSave(networkFile);
  /*
      Verif
  */
  Serial.println("\nVerification:");
  MLMatrix<float> u(Neurons[0], 1, 0.0f); // Input array
  float totalError = 0.0f;
  int nTest = 10;
  for (int i = 0; i < nTest; ++i) {
    float xx = float(random(1000)) / 1000.0f * PI;
    float yy = sin(xx);
    u(0, 0) = xx;
    MLMatrix<float> yhat = Net.predict (u);
    float pred = yhat(0, 0);
    float error = abs(yy - pred);
    Serial.printf("x= %6.3f, expected= %6.3f, predicted= %6.3f, error= %8.4f\n",
                  xx, yy, pred, error);
    totalError += error;
  }
  Serial.printf("Average error on %d verification data: %f\n", nTest, totalError / nTest);
  Net.displayNetwork();
}


void loop() {}