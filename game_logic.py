class GameState:
    def __init__(self, mode="501", players=None, double_out=True):
        self.mode = mode
        self.start_score = int(mode) if mode in ["301", "501"] else 501
        self.players = players or ["Player 1"]
        self.current_player = 0
        self.scores = {p: self.start_score for p in self.players}
        self.turn_darts = []
        self.history = []
        self.double_out = double_out

    def add_dart(self, dart_score: int, multiplier: int = 1):
        """
        Process one dart throw for the current player.
        multiplier = 1 (single), 2 (double), 3 (triple).
        """
        player = self.players[self.current_player]
        original_score = self.scores[player]
        new_score = original_score - dart_score

        # --- BUST if below 0 ---
        if new_score < 0:
            self.turn_darts.append({"dart": dart_score, "multiplier": multiplier, "status": "BUST"})
            self.scores[player] = original_score
            self.end_turn(force=True)
            return "BUST"

        # --- Check for win ---
        if new_score == 0:
            if self.double_out:
                if (multiplier == 2) or (dart_score == 50):  # Double or Bullseye
                    self.scores[player] = 0
                    self.turn_darts.append({"dart": dart_score, "multiplier": multiplier, "status": "WIN"})
                    self.history.append({"player": player, "turn": self.turn_darts})
                    return "WIN"
                else:
                    # Must finish on a double → Bust
                    self.turn_darts.append({"dart": dart_score, "multiplier": multiplier, "status": "BUST"})
                    self.scores[player] = original_score
                    self.end_turn(force=True)
                    return "BUST"
            else:
                # Normal subtract-to-zero win
                self.scores[player] = 0
                self.turn_darts.append({"dart": dart_score, "multiplier": multiplier, "status": "WIN"})
                self.history.append({"player": player, "turn": self.turn_darts})
                return "WIN"

        # --- Valid throw ---
        self.scores[player] = new_score
        self.turn_darts.append({"dart": dart_score, "multiplier": multiplier, "status": "OK"})

        # End turn if 3 darts thrown
        if len(self.turn_darts) == 3:
            self.end_turn()
        return "OK"

    def end_turn(self, force=False):
        player = self.players[self.current_player]
        self.history.append({"player": player, "turn": self.turn_darts})
        self.turn_darts = []
        if not force:
            self.next_player()

    def next_player(self):
        self.current_player = (self.current_player + 1) % len(self.players)

    def get_state(self):
        return {
            "mode": self.mode,
            "scores": self.scores,
            "current_player": self.players[self.current_player],
            "turn_darts": self.turn_darts,
            "history": self.history[-10:],
            "double_out": self.double_out
        }