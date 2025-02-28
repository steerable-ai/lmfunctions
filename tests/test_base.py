import lmfunctions as lmf


class MyDataModel(lmf.base.Base):
    attr1: str = "default1"
    attr2: str = "default2"


data = {"attr1": "value1", "attr2": "value2"}
data_str = "attr1: value1\nattr2: value2"


def test_dump():
    model = MyDataModel()
    dump = model.dump()
    assert isinstance(dump, dict)


def test_dumps():
    model = MyDataModel()
    dumps = model.dumps()
    assert isinstance(dumps, str)


def test_push(mocker):
    model = MyDataModel()
    mocker.patch("lmfunctions.base.hub.push")
    model.push("path/to/object")
    lmf.base.hub.push.assert_called_once_with("path/to/object", model.dumps())


def test_load():
    model = MyDataModel()
    model.load(data)
    assert model.attr1 == "value1"


def test_loads():
    model = MyDataModel()
    model.loads(data_str)
    assert model.attr1 == "value1"


def test_loadf(mocker):
    model = MyDataModel()
    mocker.patch("lmfunctions.base.loadf", return_value=data)
    model.loadf("url")
    lmf.base.loadf.assert_called_once_with("url")


def test_pull(mocker):
    model = MyDataModel()
    mocker.patch("lmfunctions.base.hub.pull", return_value=data)
    model.pull("path/to/object")
    lmf.base.hub.pull.assert_called_once_with("path/to/object")


def test_from_string():
    model = MyDataModel.from_string(data_str)
    assert model.attr1 == "value1"


def test_from_file(mocker):
    mocker.patch("lmfunctions.base.loadf")
    MyDataModel.from_file("url")
    lmf.base.loadf.assert_called_once_with("url")


def test_from_store(mocker):
    mocker.patch("lmfunctions.base.hub.pull", return_value=data)
    MyDataModel.from_store("path/to/object")
    lmf.base.hub.pull.assert_called_once_with("path/to/object")


def test_info():
    model = MyDataModel()
    model.info()
