import 'dart:developer';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:onnxruntime_example/tokenizers/bert_vocab.dart';
import 'package:onnxruntime_example/tokenizers/wordpiece_tokenizer.dart';
import 'model_type_test.dart';
import 'vad_iterator.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  late String _version;
  VadIterator? _vadIterator;
  static const frameSize = 64;
  static const sampleRate = 16000;

  @override
  void initState() {
    super.initState();
    _version = OrtEnv.version;
    _vadIterator = VadIterator(frameSize, sampleRate);
    _vadIterator?.initModel();
  }

  @override
  Widget build(BuildContext context) {
    const textStyle = TextStyle(fontSize: 16);
    return MaterialApp(
      theme: ThemeData(useMaterial3: true),
      home: Scaffold(
        appBar: AppBar(
          title: const Text('OnnxRuntime'),
          centerTitle: true,
        ),
        body: SingleChildScrollView(
          child: Container(
            padding: const EdgeInsets.all(10),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Text(
                  'OnnxRuntime Version = $_version',
                  style: textStyle,
                  textAlign: TextAlign.center,
                ),
                const SizedBox(
                  height: 50,
                ),
                TextButton(
                  onPressed: () async {
                    final embbedings = await testEmbedding();

                    for (var i = 0; i < embbedings.length; i++) {
                      log('embedding[$i]=${embbedings[i]}');
                    }
                  },
                  child: const Text("Test embedding"),
                ),
                TextButton(
                    onPressed: () {
                      _typeTest();
                    },
                    child: const Text('Mode Type Test')),
                const SizedBox(
                  height: 50,
                ),
                TextButton(
                    onPressed: () {
                      _vad(false);
                    },
                    child: const Text('VAD')),
                const SizedBox(
                  height: 50,
                ),
                TextButton(
                    onPressed: () {
                      _vad(true);
                    },
                    child: const Text('VAD Concurrency')),
              ],
            ),
          ),
        ),
      ),
    );
  }

  _typeTest() async {
    final startTime = DateTime.now().millisecondsSinceEpoch;
    List<OrtValue?>? outputs;
    outputs = await ModelTypeTest.testBool();
    print('out=${outputs[0]?.value}');
    for (var element in outputs) {
      element?.release();
    }
    outputs = await ModelTypeTest.testFloat();
    print('out=${outputs[0]?.value}');
    for (var element in outputs) {
      element?.release();
    }
    outputs = await ModelTypeTest.testInt64();
    print('out=${outputs[0]?.value}');
    for (var element in outputs) {
      element?.release();
    }
    outputs = await ModelTypeTest.testString();
    print('out=${outputs[0]?.value}');
    for (var element in outputs) {
      element?.release();
    }
    final endTime = DateTime.now().millisecondsSinceEpoch;
    print('infer cost time=${endTime - startTime}ms');
  }

  _vad(bool concurrent) async {
    const windowByteCount = frameSize * 2 * sampleRate ~/ 1000;
    final rawAssetFile = await rootBundle.load('assets/audio/vad_example.pcm');
    final bytes = rawAssetFile.buffer.asUint8List();
    var start = 0;
    var end = start + windowByteCount;
    List<int> frameBuffer;
    final startTime = DateTime.now().millisecondsSinceEpoch;
    while (end <= bytes.length) {
      frameBuffer = bytes.sublist(start, end).toList();
      final floatBuffer =
          _transformBuffer(frameBuffer).map((e) => e / 32768).toList();
      await _vadIterator?.predict(
          Float32List.fromList(floatBuffer), concurrent);
      start += windowByteCount;
      end = start + windowByteCount;
    }
    _vadIterator?.reset();
    final endTime = DateTime.now().millisecondsSinceEpoch;
    print('vad cost time=${endTime - startTime}ms');
  }

  Int16List _transformBuffer(List<int> buffer) {
    final bytes = Uint8List.fromList(buffer);
    return Int16List.view(bytes.buffer);
  }

  @override
  void dispose() {
    _vadIterator?.release();
    super.dispose();
  }
}

Future<List<double>> testEmbedding() {
  String texto = 'Porque Deus amou o mundo?';
  var tokens = WordpieceTokenizer(
    encoder: bertEncoder,
    decoder: bertDecoder,
    unkString: '[UNK]',
    unkToken: 100,
    startToken: 101,
    endToken: 102,
    maxInputTokens: 256,
    maxInputCharsPerWord: 100,
  ).tokenize(texto).first.tokens;
  return testTypeEmbedding(tokens, [1, 1], 'assets/models/model.onnx');
}

Future<List<double>> testTypeEmbedding(
    List tokens, List<int> shape, String assetModelName) async {
  //loading
  OrtEnv.instance.init();
  final rawAssetFile = await rootBundle.load(assetModelName);
  final bytes = rawAssetFile.buffer.asUint8List();

  final sessionOptions = OrtSessionOptions();
  final session = OrtSession.fromBuffer(bytes, sessionOptions);
  final runOptions = OrtRunOptions();
  //
  final inputOrt = OrtValueTensor.createTensorWithDataList(tokens, [1, 2]);
  final attentionMask = OrtValueTensor.createTensorWithDataList(
      List.generate(tokens.length, (index) => 1, growable: false), [1, 2]);
  final tokenTypeIds = OrtValueTensor.createTensorWithDataList(
      List.generate(tokens.length, (index) => 0, growable: false), [1, 2]);
  final inputs = {
    'input_ids': inputOrt,
    'attention_mask': attentionMask,
    'token_type_ids': tokenTypeIds
  };
  final List<OrtValue?> outputs = session.run(runOptions, inputs);

  List<double> outputValue =
      (((outputs[0]?.value as List).first as List).first as List)
          .map((e) => e as double)
          .toList();

  inputOrt.release();
  attentionMask.release();
  tokenTypeIds.release();
  //
  runOptions.release();
  sessionOptions.release();
  session.release();
  //
  OrtEnv.instance.release();
  //
  for (var i = 0; i < outputs.length; i++) {
    final element = outputs[i];
    if (element != null) {
      element.release();
    }
  }

  return outputValue;
}
