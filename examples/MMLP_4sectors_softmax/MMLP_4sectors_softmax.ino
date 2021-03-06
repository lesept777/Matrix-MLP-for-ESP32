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

void setup() {
  Serial.begin(115200);
  MLP Net(Neurons, 4, 1);
  Serial.println();
  if (!LITTLEFS.begin(FORMAT_LITTLEFS_IF_FAILED)) {
    Serial.println("LITTLEFS Mount Failed");
    return;
  }

  // Create the dataset
  int nData = 300;
  MLMatrix<float> dataX(nData, 2), dataY(nData, 4, 0);
  // draw random points in [0,1]x[0,1] and
  // set output to 0 - 3 depending on position
  for (int i = 0; i < nData; i++) {
    float xx = random(100) / 99.;
    float yy = random(100) / 99.;
    int sec = sector(xx, yy);
    dataX(i, 0) = xx;
    dataX(i, 1) = yy;
    dataY(i, sec) = 1;
  }
  Net.createDataset (dataX, dataY, nData);
  Net.setTrainTest(4, 1, 1);
  Net.normalizeDataset(dataX, dataY, 1);

  // Set parameters
  int Activations[] = {SIGMOID, SIGMOID, SOFTMAX};
  Net.setActivations(Activations);
  // Net.setCost(CROSSENTROPY);
  Net.setHyper(0.2f, 0.5f); // LR & momentum
  Net.size();


  // Training options
  long heuristics = H_INIT_OPTIM +
                    H_MUTA_WEIGH +
                    H_CHAN_LRLOG +
                    H_CHAN_SGAIN +
                    H_GRAD_CLIP  +
                    H_ZERO_WEIGH +
                    // H_DATA_SUBSE +
                    H_SELE_WEIGH;
  Net.setHeuristics(heuristics);
  //  bool initialize = !Net.netLoad(networkFile);
  //  Net.setHeurInitialize(initialize); // No need to init a new network if we read it from SPIFFS
  // Display the heuristics parameters
  Net.displayHeuristics();

  // Train...
  Net.run (dataX, dataY, 100, 10, 0.03f);
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
