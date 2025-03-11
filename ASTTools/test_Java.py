from Trans_funcUtil import invokeTransformations

code = "public class Test { public void test() { int x = 600; int y = 100; for (int i = 0; i < 10; i++) { x += y; y += x; } for (int j = 0; j < 5; j++) { y += j; } } }"
code_new = invokeTransformations(code)
print(code_new)
