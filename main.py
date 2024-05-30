# This is a Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
import numpy as np
import math
import scipy.constants as const
from scipy import integrate
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt

#  Исходные данные

rdm = pd.read_csv('rdm.csv', delimiter=';')
data = pd.read_csv('data.csv', delimiter=';')
wind_region = data['wr'][0]  # ветровой район
ter = data['ter'][0]  # тип местности
bld = data['bld'][0]  # сооружение
a = data['a'][0]  # размер здания в направлении расчетного ветра, [м]
d = data['d'][0]  # размер здания в направлении перпендикулярном расчетному направлению ветра, [м]
dz = data['dz'][0]  # отметка 0.000, [м]
eb = data['eb'][0]  # модуль упругости бетона, [МПа]
nm = data['nm'][0]  # количество учитываемых форм колебаний
c = data['c'][0]  # аэродинамический коэффициент
delta = data['delta'][0]  # логарифмический декремент
xyz = data['xyz'][0]  # расчётная поверхность
gf = 1.4  # коэффициент надёжности по нагрузке ветра
g = const.g
t11_2 = pd.read_csv('table11_2.csv', delimiter=';')
t11_4 = pd.read_csv('table11_4.csv', delimiter=';')
tksi = pd.read_csv('table_ksi.csv', delimiter=';')
t11_6 = pd.read_csv('table11_6.csv', delimiter=';')
index = ['Ia', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
table_11_1 = pd.Series([0.17, 0.23, 0.3, 0.38, 0.48, 0.6, 0.73, 0.85], index=index)


def level(n):
    """Отметки этажей:"""

    x = np.zeros(n)
    x0 = dz
    for i in range(0, n):
        x[i] = x0 + rdm['hi'][i]
        x0 = x[i]
    return x


def mass(n):
    """Матрица масс"""

    m = np.zeros((n, n))
    for i in range(0, n):
        m[i, i] = rdm['Mp'][i]
    return m


def mi(x, xi, xj, s, ei):
    """Моменты от единичных сил"""

    if x <= xi:
        m0 = s * xi - s * x
    else:
        m0 = 0
    if x <= xj:
        m1 = s * xj - s * x
    else:
        m1 = 0
    m = (m0 * m1) / ei
    return m


def m_i(x, xi, s):
    """Моменты от единичных сил"""

    if x <= xi:
        m = s * xi - s * x
    else:
        m = 0
    return m


def m_d(xi, ei, n):
    """Матрица податливости"""

    md = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            md[i, j] = integrate.quad(mi, 0, xi[n - 1], args=(xi[i], xi[j], 1, ei[i]))[0]
    return md


def ww(md):
    """Круговая частота"""

    w = math.sqrt(1 / md)
    return w


def plotmode(xi, u):
    """График форм колебаний"""

    x0 = xi
    xnew = np.linspace(x0.min(), x0.max(), 200)
    for i in range(0, nm):
        vy = np.hstack((0, u[:, i]))
        spl = make_interp_spline(x0, vy, k=3)
        y = spl(xnew)
        plt.plot(xnew, y, label='U' + str(i))
    plt.legend()
    ax = plt.gca()
    ax.set_xlabel("z, [м]", fontsize=15, color='blue')
    plt.title('Формы колебаний')
    plt.show()


def zei(h, z):
    """Эквивалентная высота"""

    ze = 0.0
    if bld == "здание":
        if h <= z:
            ze = h
        elif d < h <= 2 * d:
            if z >= h - d:
                ze = h
            elif 0 < z < h - d:
                ze = d
            else:
                ze = z
        elif h > 2 * d:
            if z >= h - d:
                ze = h
            elif d < z < h - d:
                ze = z
            elif 0 <= z <= d:
                ze = d
            else:
                ze = z
        else:
            ze = z
    return ze


def plot_zei(zi, ze_i):
    """График эквивалентной высоты"""

    plt.plot(zi, ze_i, label='$z_{e}$')
    plt.legend()
    plt.xlabel('z, [м]')
    plt.ylabel('$z_e$, [м]')
    plt.title('Эквивалентная высота $z_{e}$, м')
    plt.show()


def plot_w(w, txt, ind):
    """График ветровой нагрузки"""

    plt.plot(w, label='w' + ind)
    plt.legend()
    plt.xlabel('$z_e$, [м]')
    plt.ylabel('w' + ind + '[ кПа]')
    plt.title(txt + 'w' + ind)
    plt.show()


def ki(tb, z):
    """Интерполяция таблицы 11.2"""

    tb.index = pd.to_numeric(tb.index)
    new_index = np.unique(list(tb.index) + [z])
    new_tb = tb.reindex(new_index).interpolate(method='polynomial', order=2)
    return new_tb


def plot_ki(z_i, k):
    """График коэффициента k"""

    plt.plot(z_i, k, label='k')
    plt.legend()
    plt.xlabel('$z_{e}$, [м]')
    plt.ylabel('k')
    plt.title('Коэффициенты учитывающие изменение ветрового давления k')
    plt.show()


def k_i(h, ze_i):
    """Коэффициенты учитывающие изменение ветрового давления k"""

    tbl = t11_2.set_index(['zei'])
    i = 0
    for z in ze_i:
        tbl = ki(tbl, z)
        i = i + 1
    tb = tbl[tbl.index <= h]
    z_i = tb.index
    k = tb[ter]
    return k, z_i


def plot_zeta(z_i, zeta_i):
    """График коэффициента zeta"""

    plt.plot(z_i, zeta_i, label=r"$\zeta$")
    plt.legend()
    plt.xlabel('$z_{ei}$')
    plt.ylabel(r'$\zeta$')
    plt.title(r'Коэффициенты пульсаций давления ветра $\zeta$')
    plt.show()


def zetai(tb, z):
    """Интерполяция таблицы 11.4"""

    tb.index = pd.to_numeric(tb.index)
    new_index = np.unique(list(tb.index) + [z])
    new_tb = tb.reindex(new_index).interpolate(method='polynomial', order=2)
    return new_tb


def zeta(h, ze_i):
    """Коэффициенты пульсаций давления ветра"""

    tbl = t11_4.set_index(['zei'])
    i = 0
    for z in ze_i:
        tbl = zetai(tbl, z)
        i = i + 1
    tb = tbl[tbl.index <= h]
    zeta_i = tb[ter]
    return zeta_i


def plot_txi(table_xi):
    """Коэффициенты динамичности по рис. 11.1"""

    plt.plot(table_xi.index, table_xi['0.15'], label=r'$\delta = 0.15$')
    plt.legend()
    plt.xlabel('$T_{g,1}$')
    plt.ylabel(r'$\xi$')
    plt.plot(table_xi.index, table_xi['0.22'], label=r'$\delta = 0.22$')
    plt.legend()
    plt.ylabel(r'$\xi$')
    plt.plot(table_xi.index, table_xi['0.3'], label=r'$\delta = 0.3$')
    plt.legend()
    plt.ylabel(r'$\xi$')
    plt.title(r'$Коэффициенты динамичности \ \xi$')
    plt.show()


def plot_fl(k, fl):
    """График fl"""

    plt.plot(k, fl, label='f_lim')
    plt.legend()
    plt.xlabel('k, [м]')
    plt.ylabel('$f_{lim}$, [Гц]')
    plt.title('Предельные значения частоты собственных колебаний f_lim')
    plt.show()


def tg_1(k, f, w0):
    """Безразмерный период"""

    tg1 = math.sqrt(w0 * k * gf * 1000) / (940 * f)
    return tg1


def plot_xi_(ti, xi_):
    """График xi"""

    plt.plot(ti, xi_, label=r"$\xi$")
    plt.legend()
    plt.xlabel('$T_{g,1}$')
    plt.ylabel(r'$\xi$')
    plt.title(r'Коэффициенты динамичности для $\delta$=' + str(delta))
    plt.show()


def xii(tbl, tg1):
    """Интерполяция коэффициентов динамичности"""

    tbl.index = pd.to_numeric(tbl.index)
    new_index = np.unique(np.union1d(list(tbl.index), tg1))
    new_tb = tbl.reindex(new_index).interpolate(method='polynomial', order=2)
    return new_tb


def xi_i(tbl, tg1):
    """Коэффициенты динамичности"""

    for t in tg1:
        tbl = xii(tbl, t)
    tb = tbl[tbl.index <= max(tg1)]
    ti = tb.index
    xi_ = tb[str(delta)]
    return ti, xi_


def vi(tbl, rho):
    """Интерполяция таблицы 11_6"""

    tbl.index = pd.to_numeric(tbl.index)
    new_index = np.unique(np.union1d(list(tbl.index), rho))
    new_tb = tbl.reindex(new_index).interpolate(method='polynomial', order=2)
    return new_tb


def v_i(tbl, rho, chi):
    """Коэффициент корреляции"""

    tbl = vi(tbl, rho)
    tb = tbl[tbl.index <= 640]
    v_ = tb[str(chi)]
    return v_


def x_y_z(h):
    """Параметры коэффициента корреляции"""

    b = d
    if xyz == 'z0y':
        rho = b
        chi = h
    elif xyz == 'z0x':
        rho = 0.4 * a
        chi = h
    else:
        rho = b
        chi = a
    return rho, chi


def w_g(f, fl, wm, zet, xi, v):
    """Пульсационная составляющая основной ветровой нагрузки"""

    if f[0] >= fl:
        wg = (wm * zet) * v
    elif f[0] < fl <= f[1]:
        wg = ((wm * xi) * zet) * v
    else:
        wg = wm * 0
        print('Вторая собственная частота меньше предельной !', np.round(f[1], 4), '<', np.round(fl, 4))
    return wg


def plot_rdm(x, n):
    """Рисунок RDM"""

    img = plt.imread('rdm.png')
    fig, ax = plt.subplots()
    ax.imshow(img, extent=[0, x[n] / 1.5, 0, x[n]])
    plt.title('Расчётная динамическая модель РДМ')
    ax.axes.xaxis.set_visible(False)
    plt.show()


def calc():
    """Расчёт"""

    print('Расчёт ветровой нагрузки по СП 20.13330.2016')  # Press Ctrl+F8 to toggle the breakpoint.
    w0 = table_11_1.loc[wind_region]  # нормативное значение ветрового давления
    n = rdm.shape[0]  # количество этажей
    xi = np.hstack((0.0, level(n)))
    mm = mass(n)
    ei = eb * rdm['It'] * 1000
    md = m_d(xi, ei, n)
    dd = md @ mm
    md, u = np.linalg.eig(dd)
    vw = np.vectorize(ww)
    w = vw(np.real(np.real(md)[:nm]))
    print('Круговая частота, [рад/с]:')
    print(w[:4])
    t = 2 * math.pi / w
    print('Период, [с]:')
    print(t[:4])
    f = w / (2 * math.pi)
    print('Техническая частота, [Гц]:')
    print(f[:4])
    print('Собственные векторы:')
    print(np.real(u)[:nm, :nm])
    vzei = np.vectorize(zei)
    h = xi[n]  # высота здания от поверхности земли
    ze_i = vzei(h, xi)
    k, z_i = k_i(h, ze_i)
    wm = w0 * k * c
    zeta_i = zeta(h, ze_i)
    table_xi = tksi.set_index(['Tgi'])
    vtg_1 = np.vectorize(tg_1)
    tg1 = vtg_1(k, f[0], w0)  # безразмерный период
    ti, xi_ = xi_i(table_xi, tg1)  # коэффициенты динамичности
    if delta == 0.15:
        tglim = 0.0077
    elif delta == 0.22:
        tglim = 0.014
    else:
        tglim = 0.023
    fl = vtg_1(k, tglim, w0)
    flim = fl[k.shape[0] - 1]  # предельное значение частоты собственных колебаний
    print('Нормативное значение ветрового давления w0=', w0, ', [кПа]')
    print('Предельное значение частоты собственных колебаний f_lim=', np.round(flim, 3), ', [Гц]')
    rho, chi = x_y_z(h)
    cols = [0]
    for col in t11_6.columns:
        cols.append(col)
    chii = list(map(int, cols[2:]))
    chi1 = max(filter(lambda x: x <= chi, chii), default=None)
    chi2 = min(filter(lambda x: x >= chi, chii), default=None)
    table_v = t11_6.set_index(['rho'])
    v1 = v_i(table_v, rho, chi1)
    v2 = v_i(table_v, rho, chi2)
    v = (v1 + (v2 - v1) / (chi2 - chi1) * (chi - chi1))[d]  # коэффициент корреляции
    print('Коэффициент корреляции v=', np.round(v, 3))
    wg = w_g(f, flim, wm, zeta_i, xi_[tg1].to_numpy(), v)
    if f[1] >= flim:
        plot_w(wg, 'Норм. пульсационная составляющая основной ветровой нагрузки ', 'g')
        plot_w(wm + wg, 'Нормативное значение основной ветровой нагрузки ', '')
    return w0, n, xi, u, ze_i, k, wm, z_i, zeta_i, table_xi, ti, xi_, fl, f, flim, wg


def main():
    """Точка входа"""

    w0, n, xi, u, ze_i, k, wm, z_i, zeta_i, table_xi, ti, xi_, fl, f, flim, wg = calc()
    plot_rdm(xi, n)
    plotmode(xi, np.real(u))
    plot_zei(xi, ze_i)
    plot_ki(z_i, k)
    plot_w(wm, 'Норм. средняя составляющая основной ветровой нагрузки ', 'm')
    plot_zeta(z_i, zeta_i)
    plot_txi(table_xi)
    plot_xi_(ti, xi_)
    plot_fl(k, fl)
    if f[1] >= flim:
        plot_w(wg, 'Норм. пульсационная составляющая основной ветровой нагрузки ', 'g')
        plot_w(wm + wg, 'Нормативное значение основной ветровой нагрузки ', '')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
