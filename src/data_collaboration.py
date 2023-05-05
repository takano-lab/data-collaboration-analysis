import tqdm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD


class DataCollaborationAnalysis:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def produce_anchor(self, train):
        """
        アンカーデータを生成する関数
        """
        np.random.seed(seed=self.config["seed"])
        self.anchor = np.random.rand(self.config["num_anchor_data"], train.shape[1])
        return self.anchor

    def split_data(self, train, test, train_rating, test_rating):
        print("********************データの分割********************")
        """
        複数機関を想定してデータセットを分割する関数
        """
        # 格納しておくリスト
        train_x_list = []
        train_y_list = []
        test_x_list = []
        test_y_list = []

        # データセットを分割する
        for institute_start in tqdm.tqdm(
            range(
                0,
                self.config["num_institution"] * self.config["num_institution_user"],
                self.config["num_institution_user"],
            )
        ):
            # 一時的に格納する変数
            temp_train_x = pd.DataFrame()
            temp_test_x = pd.DataFrame()

            for user_id in range(
                institute_start, institute_start + self.config["num_institution_user"]
            ):
                user_id += 1

                # trainのuser_idが一致する箇所のindexを抽出
                train_index = train[f"uid_{user_id}"] == 1
                # testのuser_idが一致する箇所のindexを抽出
                test_index = test[f"uid_{user_id}"] == 1

                # train_index, test_indexに一致するデータを抽出
                temp_train_x = pd.concat([temp_train_x, train[train_index]], axis=0)
                temp_test_x = pd.concat([temp_test_x, test[test_index]], axis=0)

            # tempを1つのarrayに変換し、リストに格納
            train_x_list.append(temp_train_x.values)
            test_x_list.append(temp_test_x.values)

            # yはtemp_train_xに対応するratingを格納
            train_y_list.append(train_rating[temp_train_x.index])
            test_y_list.append(test_rating[temp_test_x.index])

        print("機関の数: ", len(train_x_list))

        return train_x_list, train_y_list, test_x_list, test_y_list

    def make_intermediate_expression(self, train_x_list, test_x_list, anchor):
        print("********************中間表現の生成********************")
        """
        中間表現を生成する関数
        """
        intermediate_train_x_list = []
        intermediate_test_x_list = []
        intermediate_anchor_list = []
        for institute in tqdm.tqdm(range(self.config["num_institution"])):
            # 各機関の訓練データ, テストデータおよびアンカーデータを取得し、svdを適用
            svd_train_x, svd_anchor, svd_test_x = self.svd(
                train_x_list[institute],
                anchor,
                test_x_list[institute],
                self.config["dim_intermediate"],
            )
            # svdを適用したデータをリストに格納
            intermediate_train_x_list.append(svd_train_x)
            intermediate_test_x_list.append(svd_test_x)
            intermediate_anchor_list.append(svd_anchor)

        print("中間表現の次元数: ", intermediate_train_x_list[0].shape[1])
        return (
            intermediate_train_x_list,
            intermediate_test_x_list,
            intermediate_anchor_list,
        )

    def make_integrate_expression(
        self,
        intermediate_train_x_list,
        intermediate_test_x_list,
        intermediate_anchor_list,
        train_y_list,
        test_y_list,
    ):
        print("********************統合表現の生成********************")
        """
        統合表現を生成する関数
        """
        # アンカーデータを水平方向に開く（アンカーデータ数 × 各機関の中間表現次元の合計）
        centralized_anchor = np.hstack(intermediate_anchor_list)

        # 特異値分解（Uはアンカーデータ数 × 統合表現の次元数）
        U, _, _ = np.linalg.svd(centralized_anchor)
        U = U[:, : self.config["dim_integrate"]]
        # Zは統合表現の次元数 × アンカーデータ数
        Z = U.T

        # 各機関の統合関数を求め、統合表現を生成
        integrate_train_x_list = []
        integrate_test_x_list = []

        for institute in tqdm.tqdm(range(self.config["num_institution"])):
            # 各機関のアンカーデータの中間表現を転置して、擬似逆行列を求める
            pseudo_inverse = np.linalg.pinv(intermediate_anchor_list[institute].T)

            # 各機関の統合関数を求める
            integrate_function = np.dot(Z, pseudo_inverse)

            # 統合関数で各機関の中間表現を統合
            institute_train_x_intermediate = intermediate_train_x_list[institute].T
            institute_test_x_intermediate = intermediate_test_x_list[institute].T

            institute_train_x_integrate = np.dot(
                integrate_function, institute_train_x_intermediate
            )
            institute_test_x_integrate = np.dot(
                integrate_function, institute_test_x_intermediate
            )

            # 統合表現をリストに格納
            integrate_train_x_list.append(institute_train_x_integrate.T)
            integrate_test_x_list.append(institute_test_x_integrate.T)

        print("統合表現の次元数: ", integrate_train_x_list[0].shape[1])

        # 全ての機関の統合表現をくっつけ、1つのarrayに変換
        integrate_train_x = np.vstack(integrate_train_x_list)
        integrate_test_x = np.vstack(integrate_test_x_list)

        # yもくっつける
        integrate_train_y = np.hstack(train_y_list)
        integrate_test_y = np.hstack(test_y_list)

        # logにも出力
        self.logger.info("統合表現（訓練データ）の数と次元数: {}".format(integrate_train_x.shape))
        self.logger.info("統合表現（テストデータ）の数と次元数: {}".format(integrate_test_x.shape))
        self.logger.info("統合表現（訓練データの正解）の数と次元数: {}".format(integrate_train_y.shape))
        self.logger.info("統合表現（テストデータの正解）の数と次元数: {}".format(integrate_test_y.shape))

        return integrate_train_x, integrate_test_x, integrate_train_y, integrate_test_y

    def all_apply(self, train, test, train_rating, test_rating):
        """
        データ分割、中間表現の生成、統合表現の生成を一気に行う関数
        """

        # アンカーデータの生成
        anchor = self.produce_anchor(train)

        # データの分割
        train_x_list, train_y_list, test_x_list, test_y_list = self.split_data(
            train, test, train_rating, test_rating
        )

        # 中間表現の生成
        (
            intermediate_train_x_list,
            intermediate_test_x_list,
            intermediate_anchor_list,
        ) = self.make_intermediate_expression(train_x_list, test_x_list, anchor)

        # 統合表現の生成
        (
            integrate_train_x,
            integrate_test_x,
            integrate_train_y,
            integrate_test_y,
        ) = self.make_integrate_expression(
            intermediate_train_x_list,
            intermediate_test_x_list,
            intermediate_anchor_list,
            train_y_list,
            test_y_list,
        )

        return (
            train_x_list,
            train_y_list,
            test_x_list,
            test_y_list,
            integrate_train_x,
            integrate_test_x,
            integrate_train_y,
            integrate_test_y,
        )

    @staticmethod
    def svd(train_x, anchor_x, test_x, n_components):
        """
        train_xを基準にsvdを適用し、train_x, anchor_x, test_xを次元削減する関数
        """
        svd = TruncatedSVD(n_components=n_components)
        svd.fit(train_x)
        svd_train_x = svd.transform(train_x)
        svd_anchor_x = svd.transform(anchor_x)
        svd_test_x = svd.transform(test_x)
        return svd_train_x, svd_anchor_x, svd_test_x
