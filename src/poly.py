'''
Exact polynomial arithmetic over the rationals.

Coefficients are stored from lowest to highest degree.

>>> p = Poly([1, 2, 3])
>>> print(p)
3x**2 + 2x + 1

Polynomials are just functions with special structure,
and they can be called like regular functions.

>>> p(3)
34

Because of this special structure, we can do a lot of operations that can't be done on ordinary functions.

We can compute derivatives:

>>> print(p.derivative())
6x + 2

We can check for equality:

>>> q = Poly([0, 3, 2, 1])
>>> print(q)
x**3 + 2x**2 + 3x
>>> p == q
False

We can do addition, subtraction, multiplication, and scalar exponentiation:

>>> print (p + q)
x**3 + 5x**2 + 5x + 1
>>> print (p - q)
-x**3 + x**2 - x + 1
>>> print (p * q)
3x**5 + 8x**4 + 14x**3 + 8x**2 + 3x
>>> print(p ** 2)
9x**4 + 12x**3 + 10x**2 + 4x + 1

Technically, polynomials form a *Ring*
(a type of structure studied in MATH171 Abstract Algebra),
and so they do not support division.
(If they did, they would be called a *Field*.)
'''


class Poly:
    '''
    A dense polynomial with rational coefficients.

    The field .coeffs is a list of length at least 1, from lowest to
    highest degree, with no trailing zeros. Coeffs[-1] is the leading
    coefficient.

    >>> p = Poly([1, 2, 3])
    >>> p.coeffs
    [1, 2, 3]
    >>> Poly([1, 2, 0, 0]).coeffs
    [1, 2]
    >>> Poly([]).coeffs
    [0]

    The `==` operator compares polynomials by value, and polynomials
    that represent the same function are equal regardless of how their
    coefficient lists were passed in:

    >>> Poly([1, 2, 3]) == Poly([1, 2, 3, 0, 0])
    True
    >>> Poly([1, 2]) == 1
    False

    `Poly` is a mutable value type, so it is deliberately unhashable
    and cannot be used as a dict key or a set element:

    >>> hash(Poly([1, 2, 3]))
    Traceback (most recent call last):
        ...
    TypeError: unhashable type: 'Poly'

    The `repr` is the constructor call, so `eval(repr(p)) == p` for
    every Poly. The `str` is the usual algebraic notation, with terms
    in descending degree, unit coefficients suppressed, and zero
    coefficients skipped. The zero polynomial prints as `0`:

    >>> eval(repr(Poly([1, 2, 3]))) == Poly([1, 2, 3])
    True
    >>> print(Poly([1, 2, 3]))
    3x**2 + 2x + 1
    >>> print(Poly([0]))
    0
    >>> print(Poly([5]))
    5
    >>> print(Poly([-5]))
    -5
    >>> print(Poly([1, 1]))
    x + 1
    >>> print(Poly([0, 1]))
    x
    >>> print(Poly([0, 0, -1]))
    -x**2
    >>> print(Poly([1, -1, 0, 1]))
    x**3 - x + 1
    >>> print(Poly([1, 0, 2]))
    2x**2 + 1
    '''

    def __init__(self, coeffs):
        coeffs = list(coeffs)
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()
        self.coeffs = coeffs if coeffs else [0]

    __hash__ = None

    def __repr__(self):
        return f"Poly({self.coeffs!r})"

    def __str__(self):
        if self.coeffs == [0]:
            return "0"
        terms = []
        n = len(self.coeffs) - 1
        for deg in range(n, -1, -1):
            c = self.coeffs[deg]
            if c == 0:
                continue

            sign = "-" if c < 0 else "+"
            abs_c = abs(c)

            if deg == 0:
                var = ""
            elif deg == 1:
                var = "x"
            else:
                var = f"x**{deg}"

            if deg > 0 and abs_c == 1:
                coef_str = ""
            else:
                coef_str = str(abs_c)
            terms.append((sign, coef_str + var))

        first_sign, first_body = terms[0]
        res = ("-" + first_body) if first_sign == "-" else first_body

        for sign, body in terms[1:]:
            res += f" {sign} {body}"
        return res

    def __eq__(self, other):
        if isinstance(other, Poly):
            return self.coeffs == other.coeffs
        if isinstance(other, (int, float, complex)):
            return self.coeffs == [other]
        return False

    def __call__(self, x):
        return self.eval_horner(x)

    def eval_naive(self, x):
        '''
        Evaluate by summing a_i * x**i one term at a time. The
        per-term `x ** i` recomputes a power each iteration, so the
        runtime is Theta(n) power lifts, each Theta(i) for the
        exponentiation, giving Theta(n**2) overall.

        >>> Poly([1, 2, 3]).eval_naive(2)
        17
        '''
        total = 0
        for i, c in enumerate(self.coeffs):
            total += c * x**i
        return total

    def eval_horner(self, x):
        '''
        Evaluate with Horner's rule:

            (((a_n x + a_{n-1}) x + a_{n-2}) x + ...) x + a_0

        This is Theta(n) and uses only one multiply and one add per
        coefficient. It is also numerically more stable than the
        naive evaluation when consecutive coefficients have opposite
        signs, because no large intermediate x**i is computed.

        >>> Poly([1, 2, 3]).eval_horner(2)
        17
        '''
        total = 0
        for c in reversed(self.coeffs):
            total = total * x + c
        return total

    def derivative(self):
        if len(self.coeffs) <= 1:
            return Poly([0])
        d_coeffs = [i * c for i, c in enumerate(self.coeffs)][1:]
        return Poly(d_coeffs)

    def __add__(self, other):
        if not isinstance(other, Poly):
            other = Poly([other])
        max_len = max(len(self.coeffs), len(other.coeffs))
        res = [0] * max_len
        for i in range(max_len):
            if i < len(self.coeffs):
                a = self.coeffs[i]
            else:
                a = 0
            if i < len(other.coeffs):
                b = other.coeffs[i]
            else:
                b = 0
            res[i] = a + b
        return Poly(res)

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        res = self + other
        self.coeffs = res.coeffs
        return self

    def __neg__(self):
        return Poly([-c for c in self.coeffs])

    def __sub__(self, other):
        if not isinstance(other, Poly):
            other = Poly([other])
        return self + (-other)

    def __rsub__(self, other):
        return Poly([other]) - self

    def __isub__(self, other):
        res = self - other
        self.coeffs = res.coeffs
        return self

    def mul_naive(self, other):
        '''
        Naive multiplication uses Theta(n**2) multiplications.

        >>> Poly([1, 2]) .mul_naive(Poly([3, 4]))
        Poly([3, 10, 8])
        '''
        a, b = self.coeffs, other.coeffs
        if not a or not b:
            return Poly([0])
        out = [0] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            for j, bj in enumerate(b):
                out[i + j] += ai * bj
        return Poly(out)

    def mul_karatsuba(self, other):
        if not isinstance(other, Poly):
            other = Poly([other])
        return Poly(_karatsuba(self.coeffs, other.coeffs))

    def __mul__(self, other):
        if not isinstance(other, Poly):
            return Poly([c * other for c in self.coeffs])
        return self.mul_karatsuba(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __imul__(self, other):
        res = self * other
        self.coeffs = res.coeffs
        return self

    def __pow__(self, power):
        if not isinstance(power, int) or power < 0:
            raise ValueError("Exponent must be a non-negative integer")
        res = Poly([1])
        base = self
        while power > 0:
            if power % 2 == 1:
                res = res * base
            base = base * base
            power //= 2
        return res

    def __ipow__(self, power):
        res = self ** power
        self.coeffs = res.coeffs
        return self


def _trim(coeffs):
    #need remove trailing zero coefficients as some polynomials would be equal.
    while len(coeffs) > 1 and coeffs[-1] == 0:
        coeffs.pop()
    return coeffs if coeffs else [0]


def _karatsuba(a, b):
    '''
    Product of two coefficient lists by Karatsuba's algorithm.

    The inputs are coefficient lists, lowest degree first.
    The return value is the coefficient list of the product.
    The recursion splits both operands at the midpoint and
    uses only three half-size products.
    That gives the recurrence

        T(n) = 3 T(n/2) + Theta(n)
             = Theta(n^log2 3)          # by master theorem
             = O(n^1.586)               # by defn of O()

    See <https://www.geeksforgeeks.org/dsa/multiply-two-polynomials-2/> for a basic tutorial/implementation you can adapt for your code.

    >>> _karatsuba([1, 2], [3, 4])
    [3, 10, 8]
    >>> _karatsuba([1], [1])
    [1]
    >>> _karatsuba([0], [1, 2, 3])
    [0, 0, 0]
    '''
    a = _trim(list(a))
    b = _trim(list(b))

    if not a or not b or a == [0] or b == [0]:
        return [0]

    if len(a) <= 2 or len(b) <= 2:
        out = [0] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            for j, bj in enumerate(b):
                out[i + j] += ai * bj
        return _trim(out)

    half = max(len(a), len(b)) // 2

    a0, a1 = a[:half], a[half:]
    b0, b1 = b[:half], b[half:]

    z0 = _karatsuba(a0, b0)
    z2 = _karatsuba(a1, b1)

    len_sum_a = max(len(a0), len(a1))
    sum_a = [
        (a0[i] if i < len(a0) else 0) + (a1[i] if i < len(a1) else 0)
        for i in range(len_sum_a)
    ]

    len_sum_b = max(len(b0), len(b1))
    sum_b = [
        (b0[i] if i < len(b0) else 0) + (b1[i] if i < len(b1) else 0)
        for i in range(len_sum_b)
    ]

    z1 = _karatsuba(sum_a, sum_b)

    mid_len = max(len(z1), len(z0), len(z2))
    mid = [0] * mid_len
    for i in range(mid_len):
        v1 = z1[i] if i < len(z1) else 0
        v0 = z0[i] if i < len(z0) else 0
        v2 = z2[i] if i < len(z2) else 0
        mid[i] = v1 - v0 - v2

    result = [0] * (len(a) + len(b) - 1)
    for i, v in enumerate(z0):
        if i < len(result):
            result[i] += v
    for i, v in enumerate(mid):
        if i + half < len(result):
            result[i + half] += v
    for i, v in enumerate(z2):
        if i + 2 * half < len(result):
            result[i + 2 * half] += v

    return _trim(result)
