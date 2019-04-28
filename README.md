# el-2019-baseline
2019年百度的实体链指比赛（ ccks2019，https://biendata.com/competition/ccks_2019_el/ ），一个baseline

## 模型
用BiLSTM做实体标注，然后匹配实体id。

1、标注结构是“半指针-半标注”结构，以前也曾介绍过（ https://kexue.fm/archives/5409 ， https://github.com/bojone/kg-2019-baseline ）。标注结构是自己设计的，我看了很多关系抽取的论文，没有发现类似的做法。所以，如果你基于此模型做出后的修改，最终获奖了或者发表paper什么的，烦请注明一下（其实也不是太奢望）

```
@misc{
  jianlin2019bdel,
  title={Hybrid Structure of Pointer and Ragging for Entity Recognition and Linking: A Baseline},
  author={Jianlin Su},
  year={2019},
  publisher={GitHub},
  howpublished={\url{https://github.com/bojone/el-2019-baseline}},
}
```

## 用法
`python el.py`即可。

## 结果
5个epoch左右F1应该就能到达0.63了，最后基本上F1都能跑到0.63+，自动保存F1最优的模型。

## 环境
Python 2.7 + Keras 2.2.4 + Tensorflow 1.8，其中关系最大的应该是Python 2.7了，如果你用Python 3，需要修改几行代码，至于修改哪几行，自己想办法，我不是你的debugger。

欢迎入坑Keras。人生苦短，我用Keras～

## 声明
欢迎测试、修改使用，但这是我比较早的模型，文件里边有些做法在我最新版已经被抛弃，所以以后如果发现有什么不合理的地方，不要怪我故意将大家引入歧途就行了。

欢迎跟我交流讨论，但请尽量交流一些有意义的问题，而不是debug。（如果Keras不熟悉，请先自学一个星期Keras。）

<strong>特别强调</strong>：baseline的初衷是供参赛选手测试使用，如果你已经错过了参赛日期，但想要训练数据，请自行想办法向主办方索取。我不负责提供数据下载服务。

## 链接
- https://kexue.fm
- https://keras.io