/*
  Classify points in 4 sectors
  (c) Lesept - January 2022

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
int Neurons[] = {2, 30, 20, 1};
int Activations[] = {SIGMOID, SIGMOID, RELU};
MLP Net(Neurons, 4, 1);

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println();
  if (!LITTLEFS.begin(FORMAT_LITTLEFS_IF_FAILED)) {
    Serial.println("LITTLEFS Mount Failed");
    return;
  }

  int nData = 300;
  std::vector<std::vector<float> > dataX, dataY;
  // draw random points in [0,1]x[0,1] and
  // set output to 0 - 3 depending on position
  for (int i = 0; i < nData; i++) {
    float xx = random(100) / 99.;
    float yy = random(100) / 99.;
    int sec = sector(xx, yy);
    std::vector<float> X;
    X.push_back(xx);
    X.push_back(yy);
    std::vector<float> S;
    S.push_back(sec);
    dataX.push_back(X);
    dataY.push_back(S);
  }
  Net.createDataset (dataX, dataY, nData);

  Net.setActivations(Activations);
  Net.setCost(MSE);
  Net.setHyper(0.2f, 0.05f);
  Net.size();

  //  bool initialize = true;//!Net.netLoad(networkFile);

  // Training
  long heuristics = H_INIT_OPTIM +
                    H_CHAN_WEIGH +
                    H_CHAN_LRLIN +
                    H_CHAN_SGAIN +
                    H_SELE_WEIGH;
  Net.setHeuristics(heuristics);
  // Display the heuristics parameters
  Net.displayHeuristics();

  Net.setTrainTest(4, 1, 1);
  Net.normalizeDataset(dataX, dataY, 1);
  Net.run (dataX, dataY, 100, 10, 0.025f);
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
    MLMatrix<float> yhat = Net.predict (u);
    float y = yhat(0, 0);
    int pred = (int)(y + 0.5f);
    Serial.printf ("Validation %2d: expected %d, prediction %d -->",
                   i, expected, pred);
    Serial.println((expected == pred) ? "OK" : "NOK");
    if (expected == pred) ++nok;
  }
  Serial.printf("Mean classification rate : %.2f %%\n", (100.0f * nok) / nTest);

  // Display the network
  Net.displayNetwork();
}

void loop() {
  // NOPE
}
