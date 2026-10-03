import datasets
import json
import requests
from urllib.parse import urlencode
from pathlib import Path
import os

_NAME = 'RuREBus'
_CITATION = '''
@inproceedings{rurebus,
  Address = {Moscow, Russia},
  Author = {Ivanin, Vitaly and Artemova, Ekaterina and Batura, Tatiana and Ivanov, Vladimir and Sarkisyan, Veronika and Tutubalina, Elena and Smurov, Ivan},
  Title = {RuREBus-2020 Shared Task: Russian Relation Extraction for Business},
  Booktitle = {Computational  Linguistics  and  Intellectual  Technologies:  Proceedings of the International Conference “Dialog” [Komp’iuternaia Lingvistika  i  Intellektual’nye  Tehnologii:  Trudy  Mezhdunarodnoj  Konferentsii  “Dialog”]},
  Year = {2020}
}
'''.strip()
_DESCRIPTION = 'Russian Relation Extraction for Business'
_HOMEPAGE = 'https://github.com/dialogue-evaluation/RuREBus'
_VERSION = '1.0.0'


class RuREBusBuilder(datasets.GeneratorBasedBuilder):
    base_url = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?'
    public_key = 'https://disk.yandex.ru/d/t1WakmYXlL6jBw'
    final_url = base_url + urlencode(dict(public_key=public_key))
    response = requests.get(final_url)
    raw_txt_url = response.json()['href']

    _DATA_URLS = {
        'train': 'data/train.jsonl',
        'test': 'data/test.jsonl',
    }
    _RAW_TXT_URLS = {
        'raw_txt': raw_txt_url
    }
    _TYPES_PATHS = {'ent_types': 'ent_types.txt',
                    'rel_types': 'rel_types.txt'}
    VERSION = datasets.Version(_VERSION)
    BUILDER_CONFIGS = [
        datasets.BuilderConfig('data',
                               version=VERSION,
                               description='Annotated data'),
        datasets.BuilderConfig('raw_txt',
                               version=VERSION,
                               description='Raw texts without annotations'),
        datasets.BuilderConfig('ent_types',
                               version=VERSION,
                               description='All possible entity types'),
        datasets.BuilderConfig('rel_types',
                               version=VERSION,
                               description='All possible relation types'),
    ]
    DEFAULT_CONFIG_NAME = 'data'

    def _info(self) -> datasets.DatasetInfo:
        if self.config.name == 'data':
            features = datasets.Features({
                'id': datasets.Value('int32'),
                'text': datasets.Value('string'),
                'entities': datasets.Sequence(datasets.Value('string')),
                'relations': datasets.Sequence(datasets.Value('string'))
            })
        elif self.config.name == 'raw_txt':
            features = datasets.Features({
                'region': datasets.Value('string'),
                'district': datasets.Value('string'),
                'title': datasets.Value('string'),
                'text': datasets.Value('string')
            })
        else:
            features = datasets.Features({'type': datasets.Value('string')})
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        if self.config.name == 'data':
            files = dl_manager.download(self._DATA_URLS)
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={'filepath': files['train']},
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={'filepath': files['test']},
                ),
            ]
        elif self.config.name == 'raw_txt':
            folder = dl_manager.download_and_extract(self._RAW_TXT_URLS)['raw_txt']
            return [
                datasets.SplitGenerator(
                    name='raw_txt',
                    gen_kwargs={'filepath': folder},
                )
            ]
        else:
            files = dl_manager.download(self._TYPES_PATHS)
            return [
                datasets.SplitGenerator(
                    name=self.config.name,
                    gen_kwargs={'filepath': files[self.config.name]},
                )
            ]

    def _generate_examples(self, filepath):
        if self.config.name == 'data':
            with open(filepath, encoding='utf-8') as f:
                for i, line in enumerate(f):
                    yield i, json.loads(line)
        elif self.config.name == 'raw_txt':
            path = os.path.join(filepath, 'MED_txt/unparsed_txt')
            i = 0
            for root, dirs, files in os.walk(path):
                if files:
                    root = Path(root)
                    region = root.parent.name.encode('cp437').decode('cp866')
                    district = root.name.encode('cp437').decode('cp866')
                    titles = {}
                    with open(root / 'name_dict.txt', encoding='utf-8') as f_titles:
                        for line in f_titles:
                            key, title = line.split(maxsplit=1)[1].split('_', maxsplit=1)
                            titles[key] = title.strip()
                    for file in files:
                        if file != 'name_dict.txt':
                            file = Path(file)
                            key = file.name.split('_', maxsplit=1)[0]
                            title = titles[key]
                            with open(root / file, encoding='utf-8') as f:
                                text = f.read()
                            item = {
                                'region': region,
                                'district': district,
                                'title': title,
                                'text': text
                            }
                            yield i, item
                            i += 1
        else:
            with open(filepath, encoding='utf-8') as f:
                for i, line in enumerate(f):
                    yield i, {'type': line.strip()}
