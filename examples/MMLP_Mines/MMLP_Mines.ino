/*
  Test Mines: classification of sonar signals using a neural network.  The objective is to
  discriminate between sonar signals bounced off a metal cylinder (a mine) and those bounced
  off a roughly cylindrical rock.
  The dataset has 208 lines and 61 columns, the last column is 0 = Rock, 1 = Mine
  (c) Lesept - November 2020
*/
#define FORMAT_LITTLEFS_IF_FAILED true
#include <Matrix.h>
#include <MMLP.h>
const char datasetFile[] = "/sonar.csv";
const char networkFile[] = "/Network_Mines.txt";

// Declare the network
int Neurons[] = {60, 10, 2};
MLP Net(Neurons, 3, 1);

void setup() {
  Serial.begin(115200);
  Serial.println();
  if (!LITTLEFS.begin(FORMAT_LITTLEFS_IF_FAILED)) {
    Serial.println("LITTLEFS Mount Failed");
    return;
  }

  // Load the dataset from csv file
  MLMatrix<float> x, y;
  int nData = Net.readCsvFromSpiffs(datasetFile, x, y); // Read dataset
  Net.setTrainTest(4, 1, 1); // Split dataset to training, validation and test

  // Set hyperparameters
  int Activations[] = {TANH, SIGMOID};
  Net.setActivations (Activations);
  Net.setHyper(0.25f, 0.5f); // LR & momentum

  // Define training options
  long heuristics = H_INIT_OPTIM +
                    H_SELE_WEIGH +
                    H_MUTA_WEIGH +
                    H_CHAN_LRLIN +
                    H_DATA_SUBSE +
                    H_CHAN_SGAIN;
  Net.setHeuristics(heuristics);
  //  bool initialize = !Net.netLoad(networkFile);
  //  Net.setHeurInitialize(initialize); // No need to init a new network if we read it from SPIFFS
  Net.displayHeuristics(); // Display the heuristics parameters

  // Train baby, train...
  Net.run (x, y, 600, 15, 0.05f);
  Net.netSave(networkFile);

  // Prediction
  Serial.println("\nValidation on randomly chosen data:");
  MLMatrix<float> u(Neurons[0], 1, 0.0f); // Input array
  for (int i = 0; i < 10; i++) {
    int k = random(nData);
    for (int j = 0; j < Neurons[0]; ++j) u(j, 0) = x(k, j);
    float expected = y(k, 0);
    MLMatrix<float> yhat = Net.predict(u);
    float predicted = yhat(0, 0);
    Serial.printf ("Validation %d: prediction %f, expected %f --> ",
                   i, predicted, expected);
    if (abs(predicted - expected) < 0.35f) Serial.println("OK");
    else Serial.println("NOK");
  }

  // Display the network
  Net.displayNetwork();
}

void loop() {
  // NOPE
}
