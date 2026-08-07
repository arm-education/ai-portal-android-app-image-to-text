package org.arm.learningpath.imageclassification;

import android.content.Context;
import android.text.Html;

import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

final class ClipTokenizer {
    static final int CONTEXT_LENGTH = 77;

    private static final Pattern TOKEN_PATTERN = Pattern.compile(
            "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|"
                    + "[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+",
            Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE
    );
    private static final Pattern WHITESPACE_PATTERN = Pattern.compile("\\s+");

    private final Map<String, Integer> encoder;
    private final Map<String, Integer> mergeRanks;
    private final Map<Integer, String> byteEncoder;
    private final Map<String, String> bpeCache = new HashMap<>();
    private final int startToken;
    private final int endToken;

    ClipTokenizer(Context context) throws Exception {
        encoder = loadVocabulary(context);
        mergeRanks = loadMerges(context);
        byteEncoder = createByteEncoder();
        startToken = requiredToken("<|startoftext|>");
        endToken = requiredToken("<|endoftext|>");
    }

    TokenizedPrompt tokenize(String text) {
        List<Integer> tokens = new ArrayList<>();
        tokens.add(startToken);

        Matcher matcher = TOKEN_PATTERN.matcher(clean(text));
        while (matcher.find()) {
            String encodedToken = encodeUtf8Bytes(matcher.group());
            for (String piece : bytePairEncode(encodedToken).split(" ")) {
                Integer tokenId = encoder.get(piece);
                if (tokenId != null) {
                    tokens.add(tokenId);
                }
            }
        }

        if (tokens.size() > CONTEXT_LENGTH - 1) {
            tokens = new ArrayList<>(tokens.subList(0, CONTEXT_LENGTH - 1));
        }
        tokens.add(endToken);

        long[] inputIds = new long[CONTEXT_LENGTH];
        long[] attentionMask = new long[CONTEXT_LENGTH];
        Arrays.fill(inputIds, endToken);
        for (int index = 0; index < tokens.size(); index++) {
            inputIds[index] = tokens.get(index);
            attentionMask[index] = 1;
        }
        return new TokenizedPrompt(inputIds, attentionMask);
    }

    private String clean(String text) {
        String decoded = Html.fromHtml(text, Html.FROM_HTML_MODE_LEGACY).toString();
        return WHITESPACE_PATTERN.matcher(decoded.trim()).replaceAll(" ")
                .toLowerCase(Locale.ROOT);
    }

    private String encodeUtf8Bytes(String token) {
        StringBuilder encoded = new StringBuilder();
        for (byte value : token.getBytes(StandardCharsets.UTF_8)) {
            encoded.append(byteEncoder.get(Byte.toUnsignedInt(value)));
        }
        return encoded.toString();
    }

    private String bytePairEncode(String token) {
        String cached = bpeCache.get(token);
        if (cached != null) {
            return cached;
        }

        List<String> word = new ArrayList<>();
        for (int index = 0; index < token.length(); index++) {
            word.add(String.valueOf(token.charAt(index)));
        }
        if (word.isEmpty()) {
            return token;
        }
        int lastIndex = word.size() - 1;
        word.set(lastIndex, word.get(lastIndex) + "</w>");

        while (word.size() > 1) {
            Set<String> pairs = adjacentPairs(word);
            String bestPair = null;
            int bestRank = Integer.MAX_VALUE;
            for (String pair : pairs) {
                Integer rank = mergeRanks.get(pair);
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestPair = pair;
                }
            }
            if (bestPair == null) {
                break;
            }

            int separator = bestPair.indexOf('\u0000');
            String first = bestPair.substring(0, separator);
            String second = bestPair.substring(separator + 1);
            List<String> merged = new ArrayList<>();
            int index = 0;
            while (index < word.size()) {
                if (index < word.size() - 1
                        && word.get(index).equals(first)
                        && word.get(index + 1).equals(second)) {
                    merged.add(first + second);
                    index += 2;
                } else {
                    merged.add(word.get(index));
                    index++;
                }
            }
            word = merged;
        }

        String result = String.join(" ", word);
        bpeCache.put(token, result);
        return result;
    }

    private static Set<String> adjacentPairs(List<String> word) {
        if (word.size() < 2) {
            return Collections.emptySet();
        }
        Set<String> pairs = new LinkedHashSet<>();
        for (int index = 0; index < word.size() - 1; index++) {
            pairs.add(word.get(index) + '\u0000' + word.get(index + 1));
        }
        return pairs;
    }

    private int requiredToken(String token) {
        Integer tokenId = encoder.get(token);
        if (tokenId == null) {
            throw new IllegalStateException("Tokenizer vocabulary is missing " + token);
        }
        return tokenId;
    }

    private static Map<String, Integer> loadVocabulary(Context context) throws Exception {
        String json = readAsset(context, "vocab.json");
        JSONObject object = new JSONObject(json);
        Map<String, Integer> vocabulary = new HashMap<>(object.length());
        Iterator<String> keys = object.keys();
        while (keys.hasNext()) {
            String key = keys.next();
            vocabulary.put(key, object.getInt(key));
        }
        return vocabulary;
    }

    private static Map<String, Integer> loadMerges(Context context) throws IOException {
        Map<String, Integer> ranks = new HashMap<>();
        try (InputStream input = context.getAssets().open("merges.txt");
             BufferedReader reader = new BufferedReader(
                     new InputStreamReader(input, StandardCharsets.UTF_8))) {
            String line;
            int rank = 0;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }
                String[] symbols = line.split(" ");
                if (symbols.length == 2) {
                    ranks.put(symbols[0] + '\u0000' + symbols[1], rank++);
                }
            }
        }
        return ranks;
    }

    private static String readAsset(Context context, String name) throws IOException {
        StringBuilder value = new StringBuilder();
        try (InputStream input = context.getAssets().open(name);
             BufferedReader reader = new BufferedReader(
                     new InputStreamReader(input, StandardCharsets.UTF_8))) {
            char[] buffer = new char[8192];
            int count;
            while ((count = reader.read(buffer)) != -1) {
                value.append(buffer, 0, count);
            }
        }
        return value.toString();
    }

    private static Map<Integer, String> createByteEncoder() {
        List<Integer> bytes = new ArrayList<>();
        for (int value = '!'; value <= '~'; value++) {
            bytes.add(value);
        }
        for (int value = 0x00A1; value <= 0x00AC; value++) {
            bytes.add(value);
        }
        for (int value = 0x00AE; value <= 0x00FF; value++) {
            bytes.add(value);
        }

        List<Integer> characters = new ArrayList<>(bytes);
        int extraCharacter = 0;
        for (int value = 0; value < 256; value++) {
            if (!bytes.contains(value)) {
                bytes.add(value);
                characters.add(256 + extraCharacter++);
            }
        }

        Map<Integer, String> mapping = new HashMap<>(256);
        for (int index = 0; index < bytes.size(); index++) {
            mapping.put(bytes.get(index), String.valueOf((char) characters.get(index).intValue()));
        }
        return mapping;
    }

    record TokenizedPrompt(long[] inputIds, long[] attentionMask) {
    }
}
