## このリポジトリについて
「魔方陣の生成」を 下記論文の進化戦略のアルゴリズムを元に実装したリポジトリです

[An evolutionary algorithm for magic squares](https://ieeexplore.ieee.org/abstract/document/1299763)



## Quick Start
指定の大きさの魔方陣を 100個 出力します

`$ python main.py --n <次数> --output_path <出力先のパス>`

`例) $ python main.py --n 5 --output_path ./` ( 5次魔方陣の生成 )

出力先のファイル形式

- log_ms5.csv : 5次魔方陣 が 100個
- lof_semi_ms5.csv : 5次半魔方陣

## 引数の説明
- --n : 次数 (魔方陣の1辺の長さ)
- -sf : semi-magic_square flag (半魔方陣も生成するなら True), default = False
- --output_path : 魔方陣の出力ファイルのパス指定 (必須)