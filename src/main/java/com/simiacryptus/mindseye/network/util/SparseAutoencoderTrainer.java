/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.network.util;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.layers.java.BinaryNoiseLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.java.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.java.SumReducerLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.SupervisedNetwork;
import com.simiacryptus.ref.lang.RefUtil;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;

@SuppressWarnings("serial")
public class SparseAutoencoderTrainer extends SupervisedNetwork {

  @Nullable
  public final DAGNode decoder;
  @Nullable
  public final DAGNode encoder;
  @Nullable
  public final DAGNode loss;
  @Nullable
  public final DAGNode sparsity;
  @Nullable
  public final DAGNode sparsityThrottleLayer;
  @Nullable
  public final DAGNode sumFitnessLayer;
  @Nullable
  public final DAGNode sumSparsityLayer;

  public SparseAutoencoderTrainer(@Nonnull final Layer encoder, @Nonnull final Layer decoder) {
    super(1);
    this.encoder = add(encoder, getInput(0));
    this.decoder = add(decoder, this.encoder);
    loss = add(new MeanSqLossLayer(), this.decoder, getInput(0));
    sparsity = add(new BinaryNoiseLayer(), this.encoder);
    sumSparsityLayer = add(new SumReducerLayer(), sparsity);
    LinearActivationLayer linearActivationLayer = new LinearActivationLayer();
    linearActivationLayer.setScale(0.5);
    sparsityThrottleLayer = add(linearActivationLayer.addRef(), sumSparsityLayer);
    sumFitnessLayer = add(new SumReducerLayer(), sparsityThrottleLayer, loss);
  }

  @Override
  public DAGNode getHead() {
    assert sumFitnessLayer != null;
    sumFitnessLayer.addRef();
    return sumFitnessLayer;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  SparseAutoencoderTrainer[] addRefs(@Nullable SparseAutoencoderTrainer[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter(x -> x != null).map(sparseAutoencoderTrainer -> sparseAutoencoderTrainer.addRef())
        .toArray(x -> new SparseAutoencoderTrainer[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  SparseAutoencoderTrainer[][] addRefs(@Nullable SparseAutoencoderTrainer[][] array) {
    return RefUtil.addRefs(array);
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  SparseAutoencoderTrainer addRef() {
    return (SparseAutoencoderTrainer) super.addRef();
  }
}
