package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
)

func main(){
	fmt.Println("Hello")
}

type neuralNet struct {
	config neuralNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut *mat.Dense
	bOut *mat.Dense
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



