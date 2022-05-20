/*
  Classify data above or below a sinus curve
  (c) Lesept - April 2020
*/
#include <Matrix.h>
#include <MMLP.h>
#define FORMAT_LITTLEFS_IF_FAILED true
const char networkFile[] = "/Network_HiLo.txt";

#define f(x) (0.5 + 0.3 * sin(4 * 3.14 * x));

// Declare the network
int Neurons[] = {2, 8, 1};
MLP Net(Neurons, 3, 1);

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println();
  if (!LITTLEFS.begin(FORMAT_LITTLEFS_IF_FAILED)) {
    Serial.println("LITTLEFS Mount Failed");
    return;
  }

  // Create the dataset
  int nData = 200;
  MLMatrix<float> dataX(nData, 2), dataY(nData, 1);
  // draw random points in [0,1]x[0,1] and
  // set output to 0 if under the curve or 1 if over
  for (int i = 0; i < nData; i++) {
    float x = (float)i / (nData - 1.);
    float T = f(x);
    float y = random(100) / 99.;
    dataX(i, 0) = x;
    dataX(i, 1) = y;
    dataY(i, 0) = (y > T) ? 1 : 0;
  }
  Net.createDataset (dataX, dataY, nData);
  Net.setTrainTest(4, 1, 1);

  // Set parameters
  int Activations[] = {RELU, RELU};
  Net.setActivations(Activations);
  Net.setHyper(0.2f, 0.05f); // LR, momentum
  Net.size();

  // Training options
  long heuristics = H_INIT_OPTIM +
                    H_CHAN_WEIGH +
                    H_CHAN_LRLIN +
                    H_CHAN_SGAIN +
                    H_SELE_WEIGH;
  Net.setHeuristics(heuristics);
  //  bool initialize = !Net.netLoad(networkFile);
  //  Net.setHeurInitialize(initialize); // No need to init a new network if we read it from SPIFFS
  Net.displayHeuristics(); // Display the heuristics parameters

  // Train...
  Net.run (dataX, dataY, 300, 20, 0.1f);
  Net.netSave(networkFile);

  // Prediction
  Serial.println("\nVerification:");
  MLMatrix<float> u(Neurons[0], 1, 0.0f); // Input array
  for (int i = 0; i < 10; i++) {
    float x = random(100) / 99.;
    float y = random(100) / 99.;
    float T = f(x);
    float expected = (y > T) ? 1 : 0;
    u(0, 0) = x;
    u(1, 0) = y;
    MLMatrix<float> yhat = Net.predict (u);
    float out = yhat(0, 0);
    Serial.printf ("Validation %d: prediction %f, expected %f --> ",
                   i, out, expected);
    if (abs(out - expected) < 0.5) Serial.println("OK");
    else Serial.println("NOK");
  }

  // Display the network
  Net.displayNetwork();
}

void loop() {
  // NOPE
}
