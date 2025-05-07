import React from 'react';
import Header from './header.js';
import { useState, useEffect } from 'react';
import Pagination from "react-js-pagination";
function Search_conts({list, word, total, kind, option}) {
    let [curList, setCurList] = useState([]);
    const [page, setPage] = useState(1);
    const Paging = () => {
        //page 바뀔 때마다 curList 새로 설정
        const handlePageChange = (page) => {
          setPage(page);
          let last_num = page*10;
          if (total < last_num) {
            last_num =  total;
          }
          setCurList(list.slice((page-1)*10, last_num))
        return (
          <div style={{position:'absolute', left:'50%', marginLeft:'-70px'}}>
          <Pagination
            activePage={page} // 현재 페이지
            itemsCountPerPage={10} // 한 페이지랑 보여줄 아이템 개수
            totalItemsCount={total} // 총 아이템 개수
            pageRangeDisplayed={5} // paginator의 페이지 범위
            prevPageText={"‹"} // "이전"을 나타낼 텍스트
            nextPageText={"›"} // "다음"을 나타낼 텍스트
            onChange={handlePageChange} // 페이지 변경을 핸들링하는 함수
          />
          </div>
        );
      };
    //새로고침(이동) 시 curList 설정
    useEffect(() => {
        //더보기 버튼을 클릭하기 전이면 3개(kind가 alist 혹은 plist), 더보기 버튼을 클릭 후면 10개
        {kind ==="alist" ? (setCurList(list.slice(0, 3))) : (kind ==="plist" ? (setCurList(list.slice(0, 3))) : (setCurList(list.slice((page-1)*10, page*10))))}
      }, [])
    //kind에 따른 버튼 리스트(alist는 민법, plist는 판례, article은 민법(더보기 버튼 클릭 후), precedent는 판례(더보기 버튼 클릭 후)
    const btnList = {
        alist: <p>{total > 3 && (<button className='moreBtn' type="button" onClick={() => {
            window.location.href = "/search/" + "article" + "/" + word + "/" + option+'/1';}}>더보기</button>)}</p>,
        plist: <p>{total > 3 && (<button className='moreBtn' type="button" onClick={() => {
            window.location.href = "/search/" + "precedent" + "/" + word + "/" + option+'/1';}}>더보기</button>)}</p>,
        article: <p><div><Paging/></div></p>,
        precedent: <p><div><Paging/></div></p>
    }
    return (
        <div class="div2">
            <ul>
            {curList.map((string) => {
                return(    
                <li class="left" style={{marginTop:20}}>
                {string.map((str, idx) => {
                       return (
                        <p>
                         //idx=0이면 제목, 아니면 내용
                        {idx === 0 ? (
                        str.map((s) => {
                            return(
                                  //검색 단어는 굵게
                                s === word ? (<span class = "div2-title bold">{ s }</span>) : (
                                    <span class = "div2-title">{ s }</span>))
                        })) : (typeof str === 'array' ? (
                                str.map((s) => {
                                    return(
                                          //검색 단어는 굵게
                                        s === word ? (<span class = "bold div2-conts">{ s }</span>) : (
                                            <span class = "div2-conts">{ s }</span>))
                                })): (<span class = "div2-conts">{ str }</span>)
                        )}
                        </p>
                       )})}
                </li>
                )})}
            </ul>
            {btnList[kind]}
        </div>
    )}
export default Search_conts;
