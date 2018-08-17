package main

import (
		"gonum.org/v1/gonum/mat"
	"math"
	"math/rand"
	"time"
	"os"
	"log"
	"encoding/csv"
	"fmt"
	)

func main(){
	f, err := os.Open("./train.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 7
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(rawCSVData)
}

type neuralNet struct {
	config neuralNetConfig
	weightHidden *mat.Dense
	biasHidden *mat.Dense
	weightOut *mat.Dense
	biasOut *mat.Dense
}

type neuralNetConfig struct {
	inputNeurons int
	outputNeurons int
	hiddenNeurons int
	numEpochs int
	learningRate float64
}

func newNetwork(config neuralNetConfig) *neuralNet {
	return &neuralNet{config: config}
}

func sigmoid(x float64) float64 {
	return 1.0/(1.0 + math.Exp(-x))
}

func sigmoidDerivative(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}

func (nn *neuralNet) train(x, y *mat.Dense) error {
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)
	wHidden := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, nil)
	bHidden := mat.NewDense(1, nn.config.hiddenNeurons, nil)
	wOut := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons,nil)
	bOut := mat.NewDense(1, nn.config.outputNeurons, nil)

	wHiddenRaw := wHidden.RawMatrix().Data
	bHiddenRaw := bHidden.RawMatrix().Data
	wOutRaw := wOut.RawMatrix().Data
	bOutRaw := bOut.RawMatrix().Data

	for _, param := range [][]float64 {
		wHiddenRaw,
		bHiddenRaw,
		wOutRaw,
		bOutRaw,
	}{
		for i := range param {
			param[i] = randGen.Float64()
		}
	}

	output := new(mat.Dense)
	if err := nn.backpropagate(x, y, wHidden, bHidden, wOut, bOut, output); err != nil {
		return err
	}

	// Define our trained neural network.
	nn.weightHidden = wHidden
	nn.biasHidden = bHidden
	nn.weightOut = wOut
	nn.biasOut = bOut

	return nil
}


func (nn *neuralNet) backpropagate(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {
	return nil
}