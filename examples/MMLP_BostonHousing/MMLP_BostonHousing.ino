/*
  Test Boston House prices
  Dataset found here: https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.data
  The dataset describes 13 numerical properties of houses in
  Boston suburbs and is concerned with modeling the price of
  houses in those suburbs in thousands of dollars.
  (c) Lesept - April 2020
*/
#define FORMAT_LITTLEFS_IF_FAILED true
#include <Matrix.h>
#include <MMLP.h>
const char datasetFile[] = "/Boston.csv";
const char networkFile[] = "/Network_Housing.txt";

// Declare the network
int Neurons[] = {13, 50, 30, 1};
int Activations[] = {RELU, RELU, SIGMOID};
MLP Net(Neurons, 4, 1);

void setup() {
  Serial.begin(115200);
  Serial.println();
  if (!LITTLEFS.begin(FORMAT_LITTLEFS_IF_FAILED)) {
    Serial.println("LITTLEFS Mount Failed");
    return;
  }

  // Load the dataset from csv file
  std::vector<std::vector<float> > x;
  std::vector<std::vector<float> > y;
  int nData = Net.readCsvFromSpiffs(datasetFile, x, y); // Read dataset
  Net.normalizeDataset(x, y, 1); // Move data in [0, 1] interval
  Net.setTrainTest(4, 1, 1); // Split dataset to training, validation and test

  // Set hyperparameters
  Net.setActivations (Activations);
  Net.setHyper(0.25f, 0.5f); // LR & momentum

  // Define training options
  long heuristics = H_INIT_OPTIM +
                    H_MUTA_WEIGH +
                    H_CHAN_LRLIN +
                    H_CHAN_SGAIN;
  Net.setHeuristics(heuristics);
//  bool initialize = !Net.netLoad(networkFile);
//  Net.setHeurInitialize(initialize); // No need to init a new network if we read it from SPIFFS
  Net.displayHeuristics(); // Display the heuristics parameters

  // Train baby, train...
  Net.run (x, y, 20, 15, 0.04f);
  Net.netSave(networkFile);

  // Prediction
  Serial.println("\nValidation on randomly chosen data:");
  MLMatrix<float> u(Neurons[0], 1, 0.0f); // Input array
  for (int i = 0; i < 10; i++) {
    int k = random(nData);
    for (int j = 0; j < Neurons[0]; ++j) u(j, 0) = x[k][j];
    float expected = y[k][0];
    MLMatrix<float> yhat = Net.predict(u);
    float predicted = yhat(0, 0);
    Serial.printf ("Validation %d: prediction %f, expected %f --> ",
                   i, predicted, expected);
    if (abs(predicted - expected) < 0.06) Serial.println("OK");
    else Serial.println("NOK");
  }

  // Display the network
  Net.displayNetwork();
}

void loop() {
  // NOPE
}
