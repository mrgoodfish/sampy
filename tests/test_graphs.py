import unittest

from sampy.graph.topology import BaseTopology
from sampy.graph.topology import SquareGridWithDiagTopology
from sampy.utils.decorators import sampy_class


class TestBaseTopology(unittest.TestCase):
    def test_creation_object(self):
        current_object = BaseTopology()
        self.assertTrue(hasattr(current_object, 'connections'))
        self.assertTrue(hasattr(current_object, 'weights'))
        self.assertTrue(hasattr(current_object, 'type'))
        self.assertTrue(hasattr(current_object, 'dict_cell_id_to_ind'))
        self.assertTrue(hasattr(current_object, 'time'))
        self.assertTrue(hasattr(current_object, 'on_ticker'))

        self.assertEqual(current_object.time, 0)
        self.assertEqual(current_object.dict_cell_id_to_ind, {})
        self.assertEqual(current_object.on_ticker, ['increment_time'])

    def test_increment_time(self):
        current_object = BaseTopology()
        current_object.increment_time()
        self.assertEqual(current_object.time, 1)
        current_object.increment_time()
        self.assertEqual(current_object.time, 2)

    def test_tick(self):
        current_object = BaseTopology()
        current_object.increment_time()
        self.assertEqual(current_object.time, 1)
        current_object.increment_time()
        self.assertEqual(current_object.time, 2)

        # we create an object with another method added to the 'on_ticker' list, and then test tick
        class TickTest(BaseTopology):
            def meth(self):
                self.meth = True

        current_object = TickTest()
        current_object.on_ticker.append('meth')
        current_object.tick()
        self.assertEqual(current_object.time, 1)
        self.assertTrue(current_object.meth)


class TestSquareGridWithDiagTopology(unittest.TestCase):
    def test_object_creation(self):
        # we need to turn the topology into a sampy class, in order to have the init of BaseTopology to be executed
        # first. That's why we create the following contener
        @sampy_class
        class Graph(SquareGridWithDiagTopology):
            def __init__(self, **kwargs):
                pass

        with self.assertRaises(ValueError):
            Graph()

        current_object = Graph(shape=(3, 3))
