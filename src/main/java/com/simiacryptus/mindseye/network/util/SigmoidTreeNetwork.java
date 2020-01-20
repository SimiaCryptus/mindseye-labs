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

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefMap;
import com.simiacryptus.ref.wrappers.RefSystem;
import com.simiacryptus.util.FastRandom;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Map;
import java.util.UUID;

@SuppressWarnings("serial")
public class SigmoidTreeNetwork extends DAGNetwork implements EvolvingNetwork {

  private final boolean multigate = false;
  double initialFuzzyCoeff = 1e-8;
  @Nullable
  private Layer alpha = null;
  @Nullable
  private Layer alphaBias = null;
  @Nullable
  private Layer beta = null;
  @Nullable
  private Layer betaBias = null;
  @Nullable
  private Layer gate = null;
  @Nullable
  private Layer gateBias = null;
  @Nullable
  private DAGNode head = null;
  @Nullable
  private NodeMode mode = null;
  private boolean skipChildStage = true;
  private boolean skipFuzzy = false;

  protected SigmoidTreeNetwork(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
    head = getNodeById(UUID.fromString(json.get("head").getAsString()));
    RefMap<UUID, Layer> layersById = getLayersById();
    if (json.get("alpha") != null) {
      alpha = layersById.get(UUID.fromString(json.get("alpha").getAsString()));
    }
    if (json.get("alphaBias") != null) {
      alphaBias = layersById.get(UUID.fromString(json.get("alphaBias").getAsString()));
    }
    if (json.get("beta") != null) {
      beta = layersById.get(UUID.fromString(json.get("beta").getAsString()));
    }
    if (json.get("betaBias") != null) {
      betaBias = layersById.get(UUID.fromString(json.get("betaBias").getAsString()));
    }
    if (json.get("gate") != null) {
      gate = layersById.get(UUID.fromString(json.get("gate").getAsString()));
    }
    if (json.get("gateBias") != null) {
      gate = layersById.get(UUID.fromString(json.get("gateBias").getAsString()));
    }
    setSkipChildStage(
        json.get("skipChildStage") != null ? json.get("skipChildStage").getAsBoolean() : skipChildStage());
    setSkipFuzzy(json.get("skipFuzzy") != null ? json.get("skipFuzzy").getAsBoolean() : isSkipFuzzy());
    mode = NodeMode.valueOf(json.get("mode").getAsString());
  }

  public SigmoidTreeNetwork(final Layer alpha, final Layer alphaBias) {
    super(1);
    this.alpha = alpha;
    this.alphaBias = alphaBias;
    mode = NodeMode.Linear;
  }

  @Nullable
  @Override
  public synchronized DAGNode getHead() {
    if (null == head) {
      synchronized (this) {
        if (null == head) {
          reset();
          final DAGNode input = getInput(0);
          assert getMode() != null;
          switch (getMode()) {
            case Linear:
              assert alphaBias != null;
              assert alpha != null;
              alphaBias.setFrozen(false);
              alpha.setFrozen(false);
              head = add(alpha.addRef(), add(alphaBias.addRef(), input));
              break;
            case Fuzzy: {
              assert gate != null;
              gateBias.setFrozen(false);
              gate.setFrozen(false);
              final DAGNode gateNode = add(gate.addRef(),
                  null != gateBias ? add(gateBias.addRef(), input) : input);
              assert alphaBias != null;
              assert alpha != null;
              alphaBias.setFrozen(false);
              alpha.setFrozen(false);
              LinearActivationLayer linearActivationLayer = new LinearActivationLayer();
              linearActivationLayer.setScale(2);
              SigmoidActivationLayer sigmoidActivationLayer = new SigmoidActivationLayer();
              sigmoidActivationLayer.setBalanced(false);
              Layer layer = linearActivationLayer.addRef();
              layer.freeze();
              head = add(new ProductInputsLayer(), add(alpha.addRef(), add(alphaBias.addRef(), input)),
                  add(layer.addRef(),
                      add(sigmoidActivationLayer.addRef(), gateNode)));
              break;
            }
            case Bilinear: {
              assert gate != null;
              gateBias.setFrozen(false);
              gate.setFrozen(false);
              final DAGNode gateNode = add(gate.addRef(),
                  null != gateBias ? add(gateBias.addRef(), input) : input);
              assert betaBias != null;
              assert beta != null;
              assert alphaBias != null;
              assert alpha != null;
              betaBias.setFrozen(false);
              beta.setFrozen(false);
              alphaBias.setFrozen(false);
              alpha.setFrozen(false);
              LinearActivationLayer linearActivationLayer = new LinearActivationLayer();
              linearActivationLayer.setScale(-1);
              SigmoidActivationLayer sigmoidActivationLayer = new SigmoidActivationLayer();
              sigmoidActivationLayer.setBalanced(false);
              SigmoidActivationLayer sigmoidActivationLayer1 = new SigmoidActivationLayer();
              sigmoidActivationLayer1.setBalanced(false);
              Layer layer = linearActivationLayer.addRef();
              layer.freeze();
              head = add(new SumInputsLayer(),
                  add(new ProductInputsLayer(), add(alpha.addRef(), add(alphaBias.addRef(), input)),
                      add(sigmoidActivationLayer1.addRef(), gateNode)),
                  add(new ProductInputsLayer(), add(beta.addRef(), add(betaBias.addRef(), input)),
                      add(sigmoidActivationLayer.addRef(),
                          add(layer.addRef(), gateNode))));
              break;
            }
            case Final:
              assert gate != null;
              gateBias.setFrozen(false);
              gate.setFrozen(false);
              final DAGNode gateNode = add(gate.addRef(),
                  null != gateBias ? add(gateBias.addRef(), input) : input);
              assert beta != null;
              assert alpha != null;
              LinearActivationLayer linearActivationLayer = new LinearActivationLayer();
              linearActivationLayer.setScale(-1);
              SigmoidActivationLayer sigmoidActivationLayer = new SigmoidActivationLayer();
              sigmoidActivationLayer.setBalanced(false);
              SigmoidActivationLayer sigmoidActivationLayer1 = new SigmoidActivationLayer();
              sigmoidActivationLayer1.setBalanced(false);
              Layer layer = linearActivationLayer.addRef();
              layer.freeze();
              head = add(new SumInputsLayer(),
                  add(new ProductInputsLayer(), add(alpha, input),
                      add(sigmoidActivationLayer1.addRef(), gateNode)),
                  add(new ProductInputsLayer(), add(beta, input), add(sigmoidActivationLayer.addRef(),
                      add(layer.addRef(), gateNode))));
              break;
          }
        }
      }
    }
    head.addRef();
    return head;
  }

  @Nullable
  public NodeMode getMode() {
    return mode;
  }

  public boolean isSkipFuzzy() {
    return skipFuzzy;
  }

  @Nonnull
  public SigmoidTreeNetwork setSkipFuzzy(final boolean skipFuzzy) {
    this.skipFuzzy = skipFuzzy;
    return this;
  }

  @Nonnull
  public SigmoidTreeNetwork setSkipChildStage(final boolean skipChildStage) {
    this.skipChildStage = skipChildStage;
    return this;
  }

  @Nonnull
  public static SigmoidTreeNetwork fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SigmoidTreeNetwork(json, rs);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  SigmoidTreeNetwork[] addRefs(@Nullable SigmoidTreeNetwork[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter((x) -> x != null).map(SigmoidTreeNetwork::addRef)
        .toArray((x) -> new SigmoidTreeNetwork[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  SigmoidTreeNetwork[][] addRefs(@Nullable SigmoidTreeNetwork[][] array) {
    return RefUtil.addRefs(array);
  }

  public void _free() {
    assert head != null;
    head.freeRef();
    assert alpha != null;
    alpha.freeRef();
    assert alphaBias != null;
    alphaBias.freeRef();
    assert beta != null;
    beta.freeRef();
    assert betaBias != null;
    betaBias.freeRef();
    assert gate != null;
    gate.freeRef();
    super._free();
  }

  public void copyState(@Nonnull final Layer from, @Nonnull final Layer to) {
    @Nullable final RefList<double[]> alphaState = from.state();
    @Nullable final RefList<double[]> betaState = to.state();
    assert alphaState != null;
    for (int i = 0; i < alphaState.size(); i++) {
      assert betaState != null;
      final double[] betaBuffer = betaState.get(i);
      final double[] alphaBuffer = alphaState.get(i);
      RefSystem.arraycopy(alphaBuffer, 0, betaBuffer, 0, alphaBuffer.length);
    }
  }

  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    assertConsistent();
    UUID headId = getHeadId();
    final JsonObject json = super.getJson(resources, dataSerializer);
    assert json != null;
    json.addProperty("head", headId.toString());
    if (null != alpha) {
      json.addProperty("alpha", alpha.getId().toString());
    }
    if (null != alphaBias) {
      assert alpha != null;
      json.addProperty("alphaBias", alpha.getId().toString());
    }
    if (null != beta) {
      json.addProperty("beta", beta.getId().toString());
    }
    if (null != betaBias) {
      assert beta != null;
      json.addProperty("betaBias", beta.getId().toString());
    }
    if (null != gate) {
      json.addProperty("gate", gate.getId().toString());
    }
    if (null != gateBias) {
      assert gate != null;
      json.addProperty("gateBias", gate.getId().toString());
    }
    assert getMode() != null;
    json.addProperty("mode", getMode().name());
    json.addProperty("skipChildStage", skipChildStage());
    json.addProperty("skipFuzzy", isSkipFuzzy());
    return json;
  }

  @Override
  public void nextPhase() {
    assert getMode() != null;
    switch (getMode()) {
      case Linear: {
        head = null;
        assert this.alpha != null;
        @Nonnull final FullyConnectedLayer alpha = (FullyConnectedLayer) this.alpha;
        //alphaList.weights.scale(2);
        assert alpha.outputDims != null;
        assert alpha.inputDims != null;
        gate = new FullyConnectedLayer(alpha.inputDims, multigate ? alpha.outputDims : new int[]{1});
        gateBias = new BiasLayer(alpha.inputDims);
        mode = NodeMode.Fuzzy;
        break;
      }
      case Fuzzy: {
        head = null;
        @Nullable final FullyConnectedLayer alpha = (FullyConnectedLayer) this.alpha;
        assert this.alphaBias != null;
        @Nonnull final BiasLayer alphaBias = (BiasLayer) this.alphaBias;
        assert alpha != null;
        assert alpha.outputDims != null;
        assert alpha.inputDims != null;
        FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(alpha.inputDims, alpha.outputDims);
        fullyConnectedLayer.set(() -> {
              return initialFuzzyCoeff * (FastRandom.INSTANCE.random() - 0.5);
            });
        beta = fullyConnectedLayer.addRef();
        assert alphaBias.bias != null;
        betaBias = new BiasLayer(alphaBias.bias.length());
        copyState(alpha, beta);
        copyState(alphaBias, betaBias);
        mode = NodeMode.Bilinear;
        if (isSkipFuzzy()) {
          nextPhase();
        }
        break;
      }
      case Bilinear:
        head = null;
        alpha = new SigmoidTreeNetwork(alpha, alphaBias);
        if (skipChildStage()) {
          ((SigmoidTreeNetwork) alpha).nextPhase();
        }
        beta = new SigmoidTreeNetwork(beta, betaBias);
        if (skipChildStage()) {
          ((SigmoidTreeNetwork) beta).nextPhase();
        }
        mode = NodeMode.Final;
        break;
      case Final:
        assert this.alpha != null;
        @Nonnull final SigmoidTreeNetwork alpha = (SigmoidTreeNetwork) this.alpha;
        assert this.beta != null;
        @Nonnull final SigmoidTreeNetwork beta = (SigmoidTreeNetwork) this.beta;
        alpha.nextPhase();
        beta.nextPhase();
        break;
    }
  }

  public boolean skipChildStage() {
    return skipChildStage;
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  SigmoidTreeNetwork addRef() {
    return (SigmoidTreeNetwork) super.addRef();
  }

  public enum NodeMode {
    Bilinear, Final, Fuzzy, Linear
  }

}
