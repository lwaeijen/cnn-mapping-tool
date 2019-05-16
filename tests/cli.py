#Tests for the cli
from subprocess import PIPE, Popen as popen
from unittest import TestCase

from context import cnn_implementer

tools=['cnn-frontend', 'cnn-backend', 'cnn-dse', 'cnn-opt']

class TestHelp(TestCase):
    def test_returns_usage_information(self):
        #Test for all tools
        for tool in tools:
            for flags, needle in [
                ('-h', 'usage:'),
                ('--help', 'usage:')
            ]:
                output = popen([tool, flags], stdout=PIPE).communicate()[0]
                self.assertTrue(needle in output)


class TestVersion(TestCase):
    def test_returns_version_information(self):
        for tool in tools:
            stdout, stderr = popen([tool, '--version'], stdout=PIPE, stderr=PIPE).communicate()
            self.assertTrue(cnn_implementer.__version__ in stderr)
