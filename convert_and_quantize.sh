echo "💪🏼 Download and compile Llama.cpp"
if [ ! -f "./llama.cpp/main" ]
then
  if [ ! -d "./llama.cpp/" ]
  then
  git clone https://github.com/StrikingLoo/llama.cpp
  fi
  cd llama.cpp && \
    make && \
    mv main ../api/llama
    cd ..
else
  echo "✅ Repository already cloned and compiled"
fi

echo "💬 Convert weights"
if ls models/*/*.bin 1> /dev/null 2>&1; then
    echo "✅ Weights already converted"
else
  if ls models/*/*.pth 1> /dev/null 2>&1; then
    # Convert model to ggml FP16 format
    python3 llama.cpp/convert-pth-to-ggml.py weights/7B/ 1
    # Quantize the model to 4-bits
    python3 llama.cpp/quantize.py weights/7B
  else
    echo "⚠️ Weight files do not exist. Please download them and place them inside the models folder"
  fi
fi
