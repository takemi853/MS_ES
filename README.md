## このリポジトリについて
「魔方陣の生成」を 下記論文の進化戦略のアルゴリズムを元に実装したリポジトリです。

An evolutionary algorithm for magic squares

## Quick Start
`$ python main.py --n <次数> -output_path '.'`

`例) $ python main.py --n 5` ( 5次魔方陣の生成 )

## 引数の説明
- --n : 次数 (魔方陣の1辺の長さ)
- -sf : semi-magic_square flag (半魔方陣も生成するなら True), default = False
- -output_path : 魔方陣の出力ファイルのパス指定 (必須)