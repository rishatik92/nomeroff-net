#!/bin/bash
wget -O models_ru.zip 'https://nomeroff.net.ua/datasets/autoriaNumberplateOcrRu-2020-10-12.zip'
unzip -d models/ models_ru.zip
mv models/autoriaNumberplateOcrRu-*/*  models/
rm -r models/autoriaNumberplateOcrRu*
rm models_ru.zip