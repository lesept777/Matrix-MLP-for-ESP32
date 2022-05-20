/*
  Test Iris: predict plant type using attributes
  Information: https://archive.ics.uci.edu/ml/datasets/Iris
  Dataset can be found here: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/
  (c) Lesept - May 2022
*/
#define FORMAT_LITTLEFS_IF_FAILED true
#include <Matrix.h>
#include <MMLP.h>
const char datasetFile[] = "/Iris.csv";
const char networkFile[] = "/Network_Iris.txt";

// Declare the network
int Neurons[] = {4, 10, 3};
MLP Net(Neurons, 3, 1);

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println();
  if (!LITTLEFS.begin(FORMAT_LITTLEFS_IF_FAILED)) {
    Serial.println("LITTLEFS Mount Failed");
    return;
  }

  // Set hyperparameters
  int Activations[] = {SIGMOID, SOFTMAX};
  Net.setActivations (Activations);
  Net.setHyper(0.25f, 0.5f); // LR & momentum

  // Load the dataset from csv file
  MLMatrix<float> x, y;
  int nData = Net.readCsvFromSpiffs(datasetFile, x, y); // Read dataset
  Net.setTrainTest(4, 1, 1); // Split dataset to training, validation and test

  // Define training options
  long heuristics = H_INIT_OPTIM +
                    H_SELE_WEIGH +
                    H_MUTA_WEIGH +
                    H_CHAN_LRLIN +
                    H_DATA_SUBSE +
                    H_CHAN_SGAIN;
  Net.setHeuristics(heuristics);
  //    bool initialize = !Net.netLoad(networkFile);
  //    Net.setHeurInitialize(initialize); // No need to init a new network if we read it from SPIFFS
  // Display the heuristics parameters
  Net.displayHeuristics();

  // Train baby, train...
  Net.run (x, y, 150, 15, 0.01f);
  Net.netSave(networkFile);

  // Prediction
  Serial.println("\nValidation on randomly chosen data:");
  MLMatrix<float> u(Neurons[0], 1, 0.0f); // Input array
  MLMatrix<float> truth(Neurons[2], 1, 0.0f); // Output array
  for (int i = 0; i < 10; i++) {
    int k = random(nData);
    for (int j = 0; j < Neurons[0]; ++j) u(j, 0) = x(k, j);
    for (int j = 0; j < Neurons[2]; ++j) truth(j, 0) = y(k, j);
    int predicted, expected, idy;
    MLMatrix<float> yhat = Net.predict(u);
    yhat.indexMax(predicted, idy);
    truth.indexMax(expected, idy);
    Serial.printf ("Validation %d: prediction %d, expected %d --> ",
                   i, predicted, expected);
    Serial.println((expected == predicted) ? "OK" : "NOK");
  }

  // Display the network
  Net.displayNetwork();
}

void loop() {
  // NOPE
}
