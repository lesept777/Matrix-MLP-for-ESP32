/*
  Classify points inside 3 circles
  Area 0: radius < 0.666
  Area 1 : 0.666 < radius < 1.333
  Area 2 : radius > 1.333
  (c) Lesept - April 2020
*/
#include <Matrix.h>
#include <MMLP.h>
#define FORMAT_LITTLEFS_IF_FAILED true
const char networkFile[] = "/CirclesNetwork.txt";

int area (float x, float y) {
  float R = sqrt(x * x + y * y);
  if (R < 0.666) return 0;
  if (R > 1.333) return 2;
  return 1;
}

// Declare the network
int Neurons[] = {2, 60, 40, 30, 20, 3};
MLP Net(Neurons, 6, 1);

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println();
  if (!LITTLEFS.begin(FORMAT_LITTLEFS_IF_FAILED)) {
    Serial.println("LITTLEFS Mount Failed");
    return;
  }

  // Create the dataset
  int nData = 300;
  MLMatrix<float> dataX(nData, 2), dataY(nData, 3, 0);
  // draw random points inside the circle of radius 2 and
  // set output to 0 - 2 depending on position
  for (int i = 0; i < nData; i++) {
    float rn01 = random(100) / 99.; // random number in [0-1]
    float theta = -3.14 + rn01 * 6.28;
    float R = random(100) / 50.;
    float x = R * cos(theta);
    float y = R * sin(theta);
    dataX(i, 0) = x;
    dataX(i, 1) = y;
    dataY(i, area(x, y)) = 1;
  }
  Net.createDataset (dataX, dataY, nData);
  Net.setTrainTest(4, 1, 1);
  //  Net.normalizeDataset(dataX, dataY, 1);

  // Set parameters
  int Activations[] = {TANH, TANH, TANH, TANH, SOFTMAX};
  Net.setActivations (Activations);
  Net.setHyper(0.2f, 0.03f); // LR, momentum
  Net.size();

  // Training options
  long heuristics = H_INIT_OPTIM +
                    H_MUTA_WEIGH +
                    H_SELE_WEIGH +
                    H_SHUF_DATAS +
                    H_TRAI_PRUNE +
                    H_ZERO_WEIGH +
                    H_CHAN_LRLIN;
  Net.setHeuristics(heuristics);
  bool initialize = !Net.netLoad(networkFile);
  Net.setHeurInitialize(initialize); // No need to init a new network if we read it from SPIFFS
  Net.setHeurZeroWeights(true, 0.4);
  Net.displayHeuristics(); // Display the heuristics parameters

  // Train...
  Net.run (dataX, dataY, 100, 10, 0.03f);
 // Net.netSave(networkFile);

  // Prediction for random points in the square [0-2]x[0-2]
  Serial.println();
  MLMatrix<float> u(Neurons[0], 1, 0.0f); // Input array
  for (int i = 0; i < 20; i++) {
    u(0, 0) = random(100) / 50.;
    u(1, 0) = random(100) / 50.;
    int expected = area(u(0, 0), u(1, 0));
    MLMatrix<float> out = Net.predict(u);
    int idx, idy;
    out.indexMax(idx, idy);
    Serial.printf ("Validation %2d: expected %d, prediction %d -->",
                   i, expected, idx);
    if (expected == idx) Serial.println("OK");
    else Serial.println("NOK");
  }

  // Display the network
  Net.displayNetwork();
}

void loop() {
  // NOPE
}
