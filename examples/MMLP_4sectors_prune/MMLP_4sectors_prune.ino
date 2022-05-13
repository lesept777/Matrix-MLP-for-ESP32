/*
  Classify points in 4 sectors
  (c) Lesept - April 2022

*/
#include <Matrix.h>
#include <MMLP.h>
#define FORMAT_LITTLEFS_IF_FAILED true
const char networkFile[] = "/SectorNetwork.txt";

int sector (float x, float y) {
  return (x >= 0.5) * 2 + (y >= 0.5);
  /*
     this is equivalent to:
    if (x <  0.5 && y < 0.5)  return 0;
    if (x <  0.5 && y >= 0.5) return 1;
    if (x >= 0.5 && y < 0.5)  return 2;
    if (x >= 0.5 && y >= 0.5) return 3;
  */
}

// Declare the network
int Neurons[] = {2, 30, 20, 4};
int Activations[] = {SIGMOID, SIGMOID, SOFTMAX};

void setup() {
  Serial.begin(115200);
  delay(200);
  MLP Net(Neurons, 4, 1);
  Serial.println();
  if (!LITTLEFS.begin(FORMAT_LITTLEFS_IF_FAILED)) {
    Serial.println("LITTLEFS Mount Failed");
    return;
  }

  int nData = 300;
  std::vector<std::vector<float> > dataX, dataY;
  // pick random points in [0,1]x[0,1] and
  // set output to 0 - 3 depending on position
  for (int i = 0; i < nData; i++) {
    float xx = random(100) / 99.;
    float yy = random(100) / 99.;
    int sec = sector(xx, yy);
    std::vector<float> X;
    X.push_back(xx);
    X.push_back(yy);
    std::vector<float> S;
    for (byte j = 0; j < 4; ++j) S.push_back((j == sec) ? 1 : 0); // one-hot vector
    dataX.push_back(X);
    dataY.push_back(S);
  }
  Net.createDataset (dataX, dataY, nData);

  Net.setActivations(Activations);
  // Net.setCost(CROSSENTROPY);
  Net.setHyper(0.2f, 0.5f); // LR & momentum

 // bool initialize = !Net.netLoad(networkFile);
  Net.size();

  // Training
  long heuristics = H_INIT_OPTIM +
                    H_MUTA_WEIGH +
                    H_CHAN_LRLIN +
                    H_CHAN_SGAIN +
                    H_GRAD_CLIP  +
                    H_ZERO_WEIGH +
                //    H_DATA_SUBSE +
                    H_TRAI_PRUNE +
                    H_SELE_WEIGH;
  Net.setHeuristics(heuristics);
  // Net.setHeurInitialize(initialize); // No need to init a new network if we read it from SPIFFS
  // Display the heuristics parameters
  Net.displayHeuristics();

  Net.setTrainTest(4, 1, 1);
  Net.normalizeDataset(dataX, dataY, 1);
  Net.run (dataX, dataY, 150, 10, 0.04f);
  Net.netSave(networkFile);

  // Prediction
  Serial.println("\nVerification:");
  MLMatrix<float> u(Neurons[0], 1, 0.0f); // Input array
  int nTest = 20;
  int nok = 0;
  for (int i = 0; i < nTest; i++) {
    u(0, 0) = random(100) / 99.;
    u(1, 0) = random(100) / 99.;
    int expected = sector(u(0, 0), u(1, 0));
    MLMatrix<float> yhat;
    yhat = Net.predict (u);
    int idx, idy;
    yhat.indexMax(idx, idy);
    Serial.printf ("Validation %2d: expected %d, prediction %d -->",
                   i, expected, idx);
    Serial.println((expected == idx) ? "OK" : "NOK");
    if (expected == idx) ++nok;
  }
  Serial.printf("Mean classification rate : %.2f %%\n", (100.0f * nok) / nTest);

  // Display the network
  Net.displayNetwork();
}

void loop() {
  // NOPE
}
