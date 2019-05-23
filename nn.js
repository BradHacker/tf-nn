class NeuralNetwork {
  constructor(a, b, c, h_activation, o_activation, model) {
    this.input_nodes = a;
    this.hidden_nodes = b;
    this.output_nodes = c;
    this.hidden_activation = h_activation;
    this.output_activation = o_activation;
    this.model =
      model && model instanceof tf.Sequential ? model : this.createModel();
  }

  dispose() {
    this.model.dispose();
  }

  copy() {
    return tf.tidy(() => {
      const modelCopy = this.createModel();
      const weights = this.model.getWeights();
      const weightCopies = weights.map(weight => weight.clone());
      modelCopy.setWeights(weightCopies);
      return new NeuralNetwork(
        this.input_nodes,
        this.hidden_nodes,
        this.output_nodes,
        this.hidden_activation,
        this.output_activation,
        modelCopy
      );
    });
  }

  mutate(rate) {
    tf.tidy(() => {
      const weights = this.model.getWeights();
      const mutatedWeights = weights.map(weight => {
        let tensor = weight;
        let shape = tensor.shape;
        let values = tensor.dataSync().slice();
        values = values.map(value => {
          if (Math.random() < rate) {
            return value + (Math.random() * 2 - 1) * 0.1;
          }
          return value;
        });
        return tf.tensor(values, shape);
      });
      this.model.setWeights(mutatedWeights);
    });
  }

  predict(inputs) {
    return tf.tidy(() => {
      const xs = tf.tensor2d([inputs]);
      const ys = this.model.predict(xs);
      const outputs = ys.dataSync();
      return outputs;
    });
  }

  createModel() {
    const model = tf.sequential();
    const hidden = tf.layers.dense({
      units: this.hidden_nodes,
      inputShape: [this.input_nodes],
      activation: this.hidden_activation
    });
    model.add(hidden);
    const output = tf.layers.dense({
      units: this.output_nodes,
      activation: this.output_activation
    });
    model.add(output);
    return model;
  }
}
