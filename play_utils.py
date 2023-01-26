import chess.engine
from reconchess import *
import chess.svg
from board_statistics.custom_board import CustomBoard
import json
import os


class Track_turn:
    def __init__(self, color):
        self.own_color = color

    def check_turn_color(self, color):
        if self.own_color != color:
            raise ValueError("Colors are different.")


def get_piece_list(boards, color, piece_nr):
    board_list = []
    for board in boards:
        board_list.append(board.pieces(piece_nr, color).tolist())
    return board_list


def equal_board_check(boards, color):
    # Pawn 1, Knight 2, Bishop 3, Rook 4, Queen 5, King 6
    for i in range(1, 6 + 1):
        board_list = []
        if i == 6:
            board_list = get_piece_list(boards, color, i)
            for j, board in enumerate(board_list):
                if boards[j].king(not color) is not None and board_list[0] == board:
                    continue
                else:
                    true_board_fen = boards[0].board_fen()
                    board_fen = boards[j].board_fen()
                    raise ValueError(
                        'Missing enemy King at Board, or our board differs "{}", Board fen:"{}", Zero Board fen:"{}'.format(
                            j, board_fen, true_board_fen))
        else:
            board_list = get_piece_list(boards, color, i)
            for j, board in enumerate(board_list):
                if board_list[0] == board:
                    continue
                else:
                    if board_list[0].count(True) is board.count(True) - 1 or board_list[0].count(True) is board.count(
                            True) + 1:
                        continue
                    else:
                        true_board_fen = boards[0].board_fen()
                        board_fen = boards[j].board_fen()
                        raise ValueError(
                            'With piece "{}" was a problem at Board "{}", Board fen:"{}", Zero Board fen:"{}'.format(i,
                                                                                                                     j,
                                                                                                                     board_fen,
                                                                                                                     true_board_fen))

def load_data(path, number_of_games):
    i = 0
    games_list = list()
    winner = list()
    for filename in os.listdir(path):
        if i == number_of_games:
            break
        filesize = 0
        filesize = os.path.getsize(path + filename)
        if filesize == 0:
            print(f'{filename} is empty')
            continue
        if filename.endswith(".json"):
            with open(path + filename, 'r', encoding='utf-8') as f:
                game = json.load(f)
                own_move_list = [
                    game['taken_moves']['true'][i]['value'] if game['taken_moves']['true'][i] is not None else 'None'
                    for i in range(len(game['taken_moves']['true']))]

                enemy_move_list = [
                    game['taken_moves']['false'][i]['value'] if game['taken_moves']['false'][i] is not None else 'None'
                    for i in range(len(game['taken_moves']['false']))]
                own_sense_list = [game['senses']['true'][i] for i in range(len(game['senses']['true']))]
                enemy_sense_list = [game['senses']['false'][i] for i in range(len(game['senses']['false']))]
                winner.append(game['winner_color'])

                games_list.append((own_move_list, enemy_move_list, own_sense_list, enemy_sense_list))
                i += 1
    return games_list, winner


def most_common(list_):
    if len(list_) == 0:
        return None
    return max(set(list_), key=list_.count)


def create_dict(list_):
    dict_ = {}
    for i in range(len(list_)):
        dict_[list_[i]] = list_.count(list_[i])
    return dict_

def predict_move(taken_moves: list, game_list: list, winner_list: list, color: bool):
    own_move_list = []
    enemy_move_list = []
    own_sense_list = []
    enemy_sense_list = []
    own_move_dict = {}
    enemy_move_dict = {}
    own_sense_dict = {}
    enemy_sense_dict = {}

    if color:
        # white, first move
        if len(taken_moves) == 0:
            for i, move in enumerate(game_list):
                if winner_list[i] is True:
                    own_move_list.append(game_list[i][0][len(taken_moves)])
                    enemy_move_list.append(game_list[i][1][len(taken_moves)])
                    own_sense_list.append(game_list[i][2][len(taken_moves)])
                    enemy_sense_list.append(game_list[i][3][len(taken_moves)])

            own_move_dict = create_dict(own_move_list)
            enemy_move_dict = create_dict(enemy_move_list)
            own_sense_dict = create_dict(own_sense_list)
            enemy_sense_dict = create_dict(enemy_sense_list)
            return own_move_dict, enemy_move_dict, own_sense_dict, enemy_sense_dict
        else:
            for i, move in enumerate(game_list):
                if winner_list[i] is True:
                    # if set(taken_moves).issubset(set(move[1])):
                    if taken_moves == move[0][0:len(taken_moves)]:
                        if taken_moves == move[0][0:len(taken_moves)]:

                            if not len(taken_moves) >= len(game_list[i][0]):
                                own_move_list.append(game_list[i][0][len(taken_moves)])

                            if not len(taken_moves) >= len(game_list[i][1]):
                                enemy_move_list.append(game_list[i][1][len(taken_moves)])

                            if not len(taken_moves) >= len(game_list[i][2]):
                                own_sense_list.append(game_list[i][2][len(taken_moves)])

                            if not len(taken_moves) >= len(game_list[i][3]):
                                enemy_sense_list.append(game_list[i][3][len(taken_moves)])

            own_move_dict = create_dict(own_move_list)
            enemy_move_dict = create_dict(enemy_move_list)
            own_sense_dict = create_dict(own_sense_list)
            enemy_sense_dict = create_dict(enemy_sense_list)

            return own_move_dict, enemy_move_dict, own_sense_dict, enemy_sense_dict

    else:
        if len(taken_moves) == 0:
            for i, move in enumerate(game_list):
                if winner_list[i] is False:
                    own_move_list.append(game_list[i][1][len(taken_moves)])
                    enemy_move_list.append(game_list[i][0][len(taken_moves) + 1])
                    own_sense_list.append(game_list[i][3][len(taken_moves)])
                    enemy_sense_list.append(game_list[i][2][len(taken_moves) + 1])
            own_move_dict = create_dict(own_move_list)
            enemy_move_dict = create_dict(enemy_move_list)
            own_sense_dict = create_dict(own_sense_list)
            enemy_sense_dict = create_dict(enemy_sense_list)
            return own_move_dict, enemy_move_dict, own_sense_dict, enemy_sense_dict
        else:
            for i, move in enumerate(game_list):
                if i == 5486:
                    x = i
                print(i)
                if winner_list[i] is False:
                    if taken_moves == move[1][0:len(taken_moves)]:
                        if not len(taken_moves) >= len(game_list[i][1]):
                            own_move_list.append(game_list[i][1][len(taken_moves)])

                        if not len(taken_moves) >= len(game_list[i][0]):
                            if len(game_list[i][0]) - 1 > len(taken_moves):
                                enemy_move_list.append(game_list[i][0][len(taken_moves) + 1])

                        if not len(taken_moves) >= len(game_list[i][3]):
                            own_sense_list.append(game_list[i][3][len(taken_moves)])

                        if not len(taken_moves) >= len(game_list[i][1]):
                            if len(game_list[i][2]) - 1 > len(taken_moves):
                                enemy_sense_list.append(game_list[i][2][len(taken_moves) + 1])

            own_move_dict = create_dict(own_move_list)
            enemy_move_dict = create_dict(enemy_move_list)
            own_sense_dict = create_dict(own_sense_list)
            enemy_sense_dict = create_dict(enemy_sense_list)
            return own_move_dict, enemy_move_dict, own_sense_dict, enemy_sense_dict


def illegal_castling(board, color):
    if color:
        castling_boards = []
        splits = board.fen().split('/')
        bP_split = splits[0]
        c_allowed_check = splits[-1]
        c_allowed_check = c_allowed_check.split(' ')
        c_allowed_check = c_allowed_check[-4]
        # black King_Side Caslting
        if "k" in c_allowed_check:
            if bP_split[-3:] == "k2r":
                lst = list(bP_split)
                lst[-1] = "1"
                lst[-2] = "k"
                lst[-3] = "r"
                if board.piece_at(59) is not None:
                    lst.insert(-3, '1')
                else:
                    if board.piece_at(58) is not None:
                        lst[-4] = "2"
                    else:
                        if board.piece_at(57) is not None:
                            lst[-4] = "3"
                        else:
                            if board.piece_at(56) is not None:
                                lst[-4] = "4"
                            else:
                                lst[-4] = "5"
                bP_split = "".join(lst)
                splits[0] = bP_split
                bP_fen = splits[-1]
                bP_fen_split = bP_fen.split(' ')
                bP_fen_split[1] = 'w'
                n = int(bP_fen_split[-2])
                n += 1
                bP_fen_split[-2] = str(n)
                n = int(bP_fen_split[-1])
                n += 1
                bP_fen_split[-1] = str(n)
                KQkq = bP_fen_split[2]
                KQkq = ''.join(ch for ch in KQkq if not ch.islower())
                if len(KQkq) == 0:
                    KQkq = "-"
                bP_fen_split[2] = KQkq
                bP_fen = " ".join(bP_fen_split)
                splits[-1] = bP_fen
                fen_of_needed_board = "/".join(splits)
                new_b = chess.Board(fen_of_needed_board)
                new_b = CustomBoard(new_b)
                castling_boards.append(new_b)
        # black Queen_Side Caslting
        if "q" in c_allowed_check:
            if bP_split[:3] == "r3k":
                lst = list(bP_split)
                lst[0] = "2"
                lst[1] = "k"
                lst[2] = "r"
                if board.piece_at(61) is not None:
                    lst.insert(3, '1')
                else:
                    if board.piece_at(62) is not None:
                        lst[3] = "2"
                    else:
                        if board.piece_at(63) is not None:
                            lst[3] = "3"
                        else:
                            lst[3] = "4"

                bP_split = "".join(lst)
                splits[0] = bP_split
                bP_fen = splits[-1]
                bP_fen_split = bP_fen.split(' ')
                bP_fen_split[1] = 'w'
                n = int(bP_fen_split[-2])
                n += 1
                bP_fen_split[-2] = str(n)
                n = int(bP_fen_split[-1])
                n += 1
                bP_fen_split[-1] = str(n)
                KQkq = bP_fen_split[2]
                KQkq = ''.join(ch for ch in KQkq if not ch.islower())
                if len(KQkq) == 0:
                    KQkq = "-"
                bP_fen_split[2] = KQkq
                bP_fen = " ".join(bP_fen_split)
                splits[-1] = bP_fen
                fen_of_needed_board = "/".join(splits)
                new_b = chess.Board(fen_of_needed_board)
                new_b = CustomBoard(new_b)
                castling_boards.append(new_b)
    else:
        castling_boards = []
        wP_fen_splits = board.fen().split('/')
        wP_fen = wP_fen_splits[-1]
        wP_fen_split = wP_fen.split(' ')
        wp_real_fen = wP_fen_split[0]
        c_allowed_check = wP_fen_split[-4]
        # white King-Side Caslting
        if "K" in c_allowed_check:
            if wp_real_fen[-3:] == "K2R":
                lst = list(wp_real_fen)
                lst[-1] = "1"
                lst[-2] = "K"
                lst[-3] = "R"
                if board.piece_at(3) is not None:
                    lst.insert(-3, '1')
                else:
                    if board.piece_at(2) is not None:
                        lst[-4] = "2"
                    else:
                        if board.piece_at(1) is not None:
                            lst[-4] = "3"
                        else:
                            if board.piece_at(0) is not None:
                                lst[-4] = "4"
                            else:
                                lst[-4] = "5"
                wp_fen = "".join(lst)
                wP_fen_split[0] = wp_fen
                wP_fen_split[1] = 'b'
                n = int(wP_fen_split[-2])
                n += 1
                wP_fen_split[-2] = str(n)
                KQkq = wP_fen_split[2]
                KQkq = ''.join(ch for ch in KQkq if not ch.isupper())
                if len(KQkq) == 0:
                    KQkq = "-"
                wP_fen_split[2] = KQkq
                wP_fen = " ".join(wP_fen_split)
                wP_fen_splits[-1] = wP_fen
                fen_of_needed_board = "/".join(wP_fen_splits)
                new_b = chess.Board(fen_of_needed_board)
                new_b = CustomBoard(new_b)
                castling_boards.append(new_b)
            # white Queen-Side Caslting
        if "Q" in c_allowed_check:
            if wp_real_fen[:3] == "R3K":
                lst = list(wp_real_fen)
                lst[0] = "2"
                lst[1] = "K"
                lst[2] = "R"
                if board.piece_at(5) is not None:
                    lst.insert(3, '1')
                else:
                    if board.piece_at(6) is not None:
                        lst[3] = "2"
                    else:
                        if board.piece_at(7) is not None:
                            lst[3] = "3"
                        else:
                            lst[3] = "4"
                wp_fen = "".join(lst)
                wP_fen_split[0] = wp_fen
                wP_fen_split[1] = 'b'
                n = int(wP_fen_split[-2])
                n += 1
                wP_fen_split[-2] = str(n)
                KQkq = wP_fen_split[2]
                KQkq = ''.join(ch for ch in KQkq if not ch.isupper())
                if len(KQkq) == 0:
                    KQkq = "-"
                wP_fen_split[2] = KQkq
                wP_fen = " ".join(wP_fen_split)
                wP_fen_splits[-1] = wP_fen
                fen_of_needed_board = "/".join(wP_fen_splits)
                new_b = chess.Board(fen_of_needed_board)
                new_b = CustomBoard(new_b)
                castling_boards.append(new_b)
    return castling_boards
